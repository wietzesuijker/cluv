"""Crash-safe bookkeeping for the races of `cluv submit`.

When `cluv submit` races several jobs (one per cluster, allocation or GPU type) and keeps the
first one to start, losing the SSH connection or the `cluv` process halfway through could leave
several of them running. To be able to recover from that:

- Every job of a race gets the same job name suffix (`-<race id>`), so it can be found on the
  cluster with `squeue --name` / `sacct --name` even when `sbatch` ran but its job id never made
  it back. (`sbatch --comment` would be the natural tag, but `sacct` only stores it on clusters
  with `AccountingStoreFlags=job_comment`, which fir and rorqual don't have.)
- The race is written to a local journal (`races.jsonl`, next to the job history) before any
  `sbatch`, along with each job id as it comes back, and is only marked as resolved once every
  losing job is confirmed to be gone (see `converge`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol, TypeVar

from cluv.cache import get_races_journal_path
from cluv.remote import Remote, run
from cluv.slurm import TERMINAL_JOB_STATES, parse_start

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_ERRORS = (subprocess.CalledProcessError, OSError)
"""Errors of remote commands that can go away when retried, like a dropped SSH connection (`ssh`
exits with 255) or `slurmdbd` being briefly unreachable."""


class _HasStart(Protocol):
    @property
    def start(self) -> datetime | None: ...


Racer = tuple[str, int]
"""A job of a race, as (cluster, job id)."""


def by_start(job: _HasStart) -> tuple[bool, datetime]:
    """Sort key for jobs by start time, with the jobs that haven't started last."""
    return (job.start is None, job.start or datetime.max.replace(tzinfo=timezone.utc))


def new_race_id() -> str:
    return secrets.token_hex(4)


@dataclasses.dataclass
class Race:
    """A race between jobs, as recorded in the journal."""

    id: str
    started_at: datetime
    job_names: dict[str, list[str]]
    """The job names of the racers on each cluster."""
    sbatched: list[str] = dataclasses.field(default_factory=list)
    """Clusters where `sbatch` was (or was about to be) run, so where racers may exist."""
    job_ids: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    """The job ids that `sbatch` gave back, on each cluster."""
    winner: Racer | None = None
    cancel_all: bool = False
    """Set when `cluv submit` was interrupted, so no racer should be kept."""
    resolved: bool = False

    @classmethod
    def start(cls, race_id: str, job_names: dict[str, list[str]]) -> Race:
        started_at = datetime.now(tz=timezone.utc)
        _append(
            {
                "race": race_id,
                "event": "start",
                "at": started_at.isoformat(),
                "job_names": job_names,
            }
        )
        return cls(id=race_id, started_at=started_at, job_names=job_names)

    def record_sbatch(self, cluster: str) -> None:
        self._record("sbatch", cluster=cluster)

    def record_job(self, cluster: str, job_id: int) -> None:
        self._record("job", cluster=cluster, job_id=job_id)

    def record_winner(self, winner: Racer) -> None:
        if self.winner != winner:
            self._record("winner", cluster=winner[0], job_id=winner[1])

    def record_cancel_all(self) -> None:
        self._record("cancel_all")

    def record_resolved(self) -> None:
        self._record("resolved")

    def _record(self, event: str, **data) -> None:
        _append({"race": self.id, "event": event, **data})
        self._apply(event, data)

    def _apply(self, event: str, data: dict) -> None:
        if event == "sbatch" and data["cluster"] not in self.sbatched:
            self.sbatched.append(data["cluster"])
        elif event == "job":
            self.job_ids.setdefault(data["cluster"], []).append(data["job_id"])
        elif event == "winner":
            self.winner = (data["cluster"], data["job_id"])
        elif event == "cancel_all":
            self.cancel_all = True
        elif event == "resolved":
            self.resolved = True

    def recovery_hint(self, cluster_to_remote: dict[str, Remote | None]) -> str:
        """What to run to clean up after this race by hand."""
        lines = [
            f"Race {self.id} is still open: some of its jobs may still be pending or running."
        ]
        lines.append("To cancel all of them:")
        for cluster in self.sbatched:
            scancel = f"scancel --me --name={','.join(self.job_names[cluster])}"
            local = cluster in cluster_to_remote and cluster_to_remote[cluster] is None
            lines.append(f"  {scancel}" if local else f"  ssh {cluster} {shlex.quote(scancel)}")
        lines.append("Or, to keep one of them and cancel the others: `cluv submit --resume`")
        return "\n".join(lines)


def load_races() -> dict[str, Race]:
    """Read every race from the journal, by id."""
    path = get_races_journal_path()
    if not path.exists():
        return {}
    races: dict[str, Race] = {}
    for line in path.read_text().splitlines():
        try:
            data = json.loads(line)
            race_id, event = data.pop("race"), data.pop("event")
            if event == "start":
                races[race_id] = Race(
                    id=race_id,
                    started_at=datetime.fromisoformat(data["at"]),
                    job_names=data["job_names"],
                )
            elif race_id in races:
                races[race_id]._apply(event, data)
        except (ValueError, KeyError, TypeError, AttributeError):
            # Most likely a line cut short by a crash while writing it.
            logger.debug(f"Skipping unreadable line in {path}: {line!r}")
    return races


def _append(record: dict) -> None:
    path = get_races_journal_path()
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        # This is a write-ahead log: make sure it hits the disk before we run `sbatch`.
        os.fsync(f.fileno())


async def retrying(function: Callable[[], Awaitable[T]], *, what: str, max_attempts: int = 8) -> T:
    """Call `function` until it doesn't raise a transient error, with exponential backoff.

    Gives up (re-raising the error) after `max_attempts` attempts, about 1.5 minutes.
    """
    delay = 1
    for attempt in range(1, max_attempts + 1):
        try:
            return await function()
        except TRANSIENT_ERRORS as err:
            if attempt == max_attempts:
                raise
            logger.warning(
                f"{what} failed ({err}). Retrying in {delay}s ({attempt}/{max_attempts})."
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise AssertionError("unreachable")


@dataclasses.dataclass(frozen=True)
class RacerState:
    cluster: str
    job_id: int
    state: str
    start: datetime | None

    @property
    def racer(self) -> Racer:
        return (self.cluster, self.job_id)

    @property
    def live(self) -> bool:
        return not self.state.startswith(tuple(TERMINAL_JOB_STATES))


async def find_racers(race: Race, cluster: str, remote: Remote | None) -> list[RacerState]:
    """Find the jobs of the race on `cluster` by their name.

    Uses `sacct`, plus `squeue` for the jobs so recent that `sacct` doesn't list them yet.
    """
    names = ",".join(race.job_names[cluster])
    since = (race.started_at - timedelta(days=1)).date().isoformat()
    sacct = await _get_output(
        remote,
        ["sacct", f"--name={names}", f"--starttime={since}", "--allocations", "--noheader",
         "--parsable2", "--format=JobID,State,Start"],
    )  # fmt: skip
    squeue = await _get_output(
        remote, ["squeue", "--me", f"--name={names}", "--noheader", "--format=%i|%T|%S"]
    )
    racers: dict[int, RacerState] = {}
    for line in [*sacct.splitlines(), *squeue.splitlines()]:
        job_id, state, start = (line.split("|") + ["", ""])[:3]
        # Job arrays show up as `<job id>_<task id>`.
        racer = RacerState(
            cluster, int(job_id.split("_")[0]), state.split()[0], parse_start(start)
        )
        racers[racer.job_id] = racer
    return list(racers.values())


async def find_all_racers(
    race: Race, cluster_to_remote: dict[str, Remote | None]
) -> list[RacerState]:
    racers = await asyncio.gather(
        *(
            retrying(
                lambda cluster=cluster: find_racers(race, cluster, cluster_to_remote[cluster]),
                what=f"Looking for the jobs of race {race.id} on {cluster}",
            )
            for cluster in race.sbatched
        )
    )
    return [racer for cluster_racers in racers for racer in cluster_racers]


async def converge(
    race: Race,
    cluster_to_remote: dict[str, Remote | None],
    winner: Racer | None,
    wait: bool = True,
) -> Racer | None:
    """Cancel every live job of the race but one, wait until they're gone, and mark the race
    as resolved.

    The job that is kept is `winner` if given (or recorded in the journal). Otherwise, it is the
    first job to have started, or the only one still pending. While several jobs are pending and
    none have started, this keeps waiting, like `cluv submit` does, unless `wait` is False: then it
    returns None and leaves the race open. No job is kept if the race was interrupted.

    Returns the job that was kept, if any. Raises one of `TRANSIENT_ERRORS` if a cluster stays
    unreachable, leaving the race open.
    """
    winner = winner or race.winner
    delay = 1
    while True:
        racers = await find_all_racers(race, cluster_to_remote)
        live = [r for r in racers if r.live]
        if race.cancel_all:
            winner = None
        elif winner is None:
            started = [r for r in racers if r.state == "COMPLETED"] or [
                r for r in racers if r.state == "RUNNING"
            ]
            if not started and len(live) > 1:
                if not wait:
                    return None
                logger.debug(
                    f"{len(live)} jobs of race {race.id} are pending, waiting for one to start."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if started or live:
                winner = min(started or live, key=by_start).racer
        if winner is not None:
            race.record_winner(winner)
        losers = [r for r in live if r.racer != winner]
        if not losers:
            race.record_resolved()
            return winner
        await asyncio.gather(
            *(
                retrying(
                    lambda cluster=cluster, ids=ids: _get_output(
                        cluster_to_remote[cluster], ["scancel", *map(str, ids)]
                    ),
                    what=f"Cancelling jobs {ids} on {cluster}",
                )
                for cluster, ids in _group_ids_by_cluster(losers).items()
            )
        )
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)


def _group_ids_by_cluster(racers: list[RacerState]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for racer in racers:
        grouped.setdefault(racer.cluster, []).append(racer.job_id)
    return grouped


async def _get_output(remote: Remote | None, command: list[str]) -> str:
    login_command = f"bash --login -c {shlex.quote(shlex.join(command))}"
    if remote is not None:
        return await remote.get_output(login_command, hide=True)
    return (await run(tuple(shlex.split(login_command)), hide=True)).stdout.strip()
