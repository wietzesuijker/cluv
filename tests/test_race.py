"""Tests for the crash-safety of the races of `cluv submit` (see `cluv/cli/submit_utils/race.py`).

The cluster is faked by `FakeSlurm`, which keeps track of the jobs and can lose its connection at
any point, to check that no racer is left behind.
"""

import asyncio
import copy
import datetime
import importlib
import re
import shlex
import subprocess
import textwrap
import unittest.mock
from pathlib import Path

import pytest

import cluv.cache
import cluv.cli.submit
import cluv.cli.submit_utils.race
import cluv.remote
import cluv.slurm
import cluv.utils
from cluv.cli.submit import ensure_clean_git_state, resume_races, submit
from cluv.cli.submit_utils.race import Race, load_races, retrying
from cluv.slurm import TERMINAL_JOB_STATES
from cluv.utils import console, current_cluster

pytestmark = pytest.mark.timeout(10)

sync_module = importlib.import_module("cluv.cli.sync")

CLUSTER = "narval"
START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


class FakeSlurm:
    """A fake Slurm cluster, reached through `cluv.remote.run`, whose connection can drop."""

    def __init__(self, current_cluster: unittest.mock.Mock) -> None:
        self.current_cluster = current_cluster
        """Make this return None to submit from (or resume on) a machine with no connection."""
        self.jobs: dict[int, dict] = {}
        self.connected = True
        self.drop_on: str | None = None
        """Lose the connection at the first command containing this."""
        self.sbatch_runs_while_dropping = False
        """Whether the `sbatch` that drops the connection still submits its job."""
        self.blip = False
        """Whether the connection comes back right after the command that dropped it."""
        self.states: list[str] = []
        """The states that the next submitted jobs are in right away (RUNNING by default)."""

    def alive(self) -> list[int]:
        return [i for i, job in self.jobs.items() if job["state"] in ("PENDING", "RUNNING")]

    async def run(self, program_and_args: tuple[str, ...], warn: bool = False, **kwargs):
        command = shlex.join(program_and_args)
        if self.drop_on and self.drop_on in command:
            self.drop_on = None
            if "sbatch" in command and self.sbatch_runs_while_dropping:
                self._sbatch(command)
            self.connected = self.blip
            if warn:
                return subprocess.CompletedProcess(program_and_args, 255, "", "Broken pipe")
            raise subprocess.CalledProcessError(255, program_and_args, "", "Broken pipe")
        if not self.connected:
            if warn:
                return subprocess.CompletedProcess(program_and_args, 255, "", "Broken pipe")
            raise subprocess.CalledProcessError(255, program_and_args, "", "Broken pipe")
        return subprocess.CompletedProcess(program_and_args, 0, self._respond(command), "")

    def _sbatch(self, command: str) -> int:
        job_id = 100 + len(self.jobs)
        name = re.search(r"--job-name=(\S+)", command)
        assert name
        # Jobs start right away, in the order in which they were submitted.
        start = START + datetime.timedelta(seconds=len(self.jobs))
        state = self.states.pop(0) if self.states else "RUNNING"
        self.jobs[job_id] = {"name": name.group(1), "state": state, "start": start}
        return job_id

    def _row(self, job_id: int) -> str:
        job = self.jobs[job_id]
        start = job["start"].strftime("%Y-%m-%dT%H:%M:%S") if job["start"] else "Unknown"
        return f"{job_id}|{job['state']}|{start}"

    def _respond(self, command: str) -> str:
        if "sbatch --parsable" in command:
            return str(self._sbatch(command))
        if match := re.search(r"sacct -j ([\d,]+) ", command):
            ids = [int(i) for i in match.group(1).split(",")]
            return "\n".join(self._row(i) for i in ids if i in self.jobs)
        if match := re.search(r"--name=(\S+?)'? ", command):
            names = match.group(1).split(",")
            ids = [i for i, job in self.jobs.items() if job["name"] in names]
            if command.startswith("bash --login -c 'squeue"):
                ids = [i for i in ids if i in self.alive()]
            return "\n".join(self._row(i) for i in ids)
        if match := re.match(r"(?:bash --login -c ')?scancel ([\d ]+)'?$", command):
            for job_id in map(int, match.group(1).split()):
                if job_id in self.alive():
                    self.jobs[job_id]["state"] = "CANCELLED"
            return ""
        pytest.fail(f"Unexpected command: {command}")


@pytest.fixture
def slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    """Submit from `CLUSTER`, which has two allocations, so every `cluv submit` races two jobs."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    (project_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""\
            [tool.cluv]
            results_path = "results"
            [tool.cluv.clusters.{CLUSTER}]
            sbatch_args = [{{ account = "rrg-bengioy-ad" }}, {{ account = "def-bengioy" }}]
            """
        )
    )
    (project_dir / "job.sh").write_text("#!/bin/bash\necho Hello World\n")

    current_cluster_mock = unittest.mock.Mock(spec=current_cluster, return_value=CLUSTER)
    monkeypatch.setattr(cluv.utils, current_cluster.__name__, current_cluster_mock)
    monkeypatch.setattr(sync_module, current_cluster.__name__, current_cluster_mock)
    monkeypatch.setattr(
        sync_module,
        sync_module.get_active_remotes.__name__,
        unittest.mock.AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        cluv.cli.submit, ensure_clean_git_state.__name__, lambda **kwargs: "dummy_git_commit"
    )
    monkeypatch.setattr(cluv.cache, cluv.cache._get_cache_dir.__name__, lambda: tmp_path)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _: real_sleep(0))

    fake = FakeSlurm(current_cluster_mock)
    for module in (cluv.remote, cluv.slurm, cluv.cli.submit, cluv.cli.submit_utils.race):
        monkeypatch.setattr(module, cluv.remote.run.__name__, fake.run)
    return fake


async def cluv_submit():
    return await submit(
        cluster=CLUSTER,
        job_script=Path("job.sh"),
        sbatch_args=[],
        program_args=[],
        _skip_sync=True,
    )


def only_race() -> Race:
    (race,) = load_races().values()
    return race


async def test_race_without_connection_loss(slurm: FakeSlurm) -> None:
    job = await cluv_submit()
    assert job is not None
    assert slurm.alive() == [job.job_id]
    race = only_race()
    assert race.resolved
    assert race.winner == (CLUSTER, job.job_id)
    assert job.sbatch_args["job-name"] == f"cluv-job-{race.id}"


@pytest.mark.parametrize("state", [s for s in TERMINAL_JOB_STATES if s != "COMPLETED"])
async def test_race_ends_when_every_job_failed(slurm: FakeSlurm, state: str) -> None:
    slurm.states = [state, state]
    assert await cluv_submit() is None
    assert only_race().resolved


@pytest.mark.parametrize("state", TERMINAL_JOB_STATES)
async def test_losing_job_that_already_ended_is_not_waited_for(
    slurm: FakeSlurm, state: str
) -> None:
    # The first job ends before the second one starts, so the second one has to be waited for.
    slurm.states = [state, "RUNNING"]
    job = await cluv_submit()
    assert job is not None
    assert only_race().resolved


async def test_job_whose_id_was_lost_is_cancelled(slurm: FakeSlurm) -> None:
    """`sbatch` ran, but the connection dropped (for a moment) before its job id came back."""
    slurm.drop_on = "--account=rrg-bengioy-ad"
    slurm.sbatch_runs_while_dropping = True
    slurm.blip = True

    job = await cluv_submit()

    # Only the `def-` job got an id back, so it wins, and the other is found by name.
    assert job is not None
    assert job.sbatch_args["account"] == "def-bengioy"
    assert len(slurm.jobs) == 2
    assert slurm.alive() == [job.job_id]
    assert only_race().resolved


async def test_connection_lost_while_polling_leaves_the_race_open(slurm: FakeSlurm) -> None:
    slurm.drop_on = "sacct -j"
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        await cluv_submit()
    assert exc_info.value.code == 1
    race = only_race()
    assert not race.resolved
    assert sorted(race.job_ids[CLUSTER]) == sorted(slurm.jobs)
    assert len(slurm.alive()) == 2  # Nothing could be cancelled.
    assert f"scancel --me --name=cluv-job-{race.id}" in capture.get()


@pytest.mark.parametrize(
    ("drop_on", "sbatch_runs", "jobs_left"),
    [
        pytest.param("sbatch --parsable", False, 0, id="before_sbatch"),
        pytest.param("--account=rrg-bengioy-ad", True, 1, id="after_sbatch_before_job_id"),
        pytest.param("sacct -j", False, 1, id="while_polling"),
        pytest.param("scancel", False, 1, id="while_cancelling"),
    ],
)
async def test_resume_after_connection_loss(
    slurm: FakeSlurm, drop_on: str, sbatch_runs: bool, jobs_left: int
) -> None:
    slurm.drop_on = drop_on
    slurm.sbatch_runs_while_dropping = sbatch_runs
    with pytest.raises(SystemExit) as exc_info:
        await cluv_submit()
    assert exc_info.value.code == 1
    assert slurm.drop_on is None  # The connection was lost at that point.
    assert not only_race().resolved

    slurm.connected = True
    await resume_races()

    race = only_race()
    assert race.resolved
    assert len(slurm.alive()) == jobs_left
    if jobs_left:
        assert race.winner == (CLUSTER, slurm.alive()[0])

    # Resuming again changes nothing.
    jobs = copy.deepcopy(slurm.jobs)
    journal = cluv.cache.get_races_journal_path().read_text()
    await resume_races()
    assert slurm.jobs == jobs
    assert cluv.cache.get_races_journal_path().read_text() == journal


async def test_resume_leaves_the_races_on_disconnected_clusters_open(slurm: FakeSlurm) -> None:
    slurm.drop_on = "sacct -j"
    with pytest.raises(SystemExit):
        await cluv_submit()
    slurm.connected = True

    # Not on the cluster anymore, and not connected to it: nothing to do but wait.
    slurm.current_cluster.return_value = None
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        await resume_races()
    assert exc_info.value.code == 1
    assert f"cluv login {CLUSTER}" in " ".join(capture.get().split())  # Undo the line wrapping.
    assert not only_race().resolved
    assert len(slurm.alive()) == 2

    slurm.current_cluster.return_value = CLUSTER
    await resume_races()
    assert only_race().resolved
    assert len(slurm.alive()) == 1


async def test_retrying_recovers_from_a_transient_error() -> None:
    function = unittest.mock.AsyncMock(
        side_effect=[subprocess.CalledProcessError(255, "ssh"), "done"]
    )
    with unittest.mock.patch.object(asyncio, "sleep", unittest.mock.AsyncMock()):
        assert await retrying(function, what="test") == "done"
    assert function.await_count == 2


async def test_retrying_gives_up() -> None:
    function = unittest.mock.AsyncMock(side_effect=subprocess.CalledProcessError(255, "ssh"))
    with unittest.mock.patch.object(asyncio, "sleep", unittest.mock.AsyncMock()):
        with pytest.raises(subprocess.CalledProcessError):
            await retrying(function, what="test", max_attempts=3)
    assert function.await_count == 3


def test_journal_survives_a_line_cut_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cluv.cache, cluv.cache._get_cache_dir.__name__, lambda: tmp_path)
    race = Race.start("abcd1234", job_names={CLUSTER: ["cluv-job-abcd1234"]})
    race.record_sbatch(CLUSTER)
    race.record_job(CLUSTER, 123)
    with cluv.cache.get_races_journal_path().open("a") as f:
        f.write('{"race": "abcd1234", "event": "wi')  # A crash while writing.

    (loaded,) = load_races().values()
    assert loaded == race
    assert not loaded.resolved
    # The job history is kept apart, so older versions of cluv don't drop the races.
    assert not (tmp_path / "jobs.jsonl").exists()
