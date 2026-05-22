"""`cluv probe`: cold-start a memory profile by submitting one throwaway job."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from cluv import history
from cluv.cli.submit import (
    _wait_terminal,
    ensure_clean_git_state,
    get_elapsed_s,
    get_max_rss_mb,
    sbatch,
)
from cluv.cli.sync import sync
from cluv.config import EstimateConfig
from cluv.utils import console

logger = logging.getLogger(__name__)

__all__ = ["probe"]

# A probe that runs past this fraction of its asked `--time` is suspicious: the
# user's script likely isn't honoring `CLUV_PROBE=1` and the probe just paid
# for a full run.
PROBE_WALL_FRAC = 0.5


async def probe(
    cluster: str,
    job_script: Path,
    mem: str,
    time: str,
    cpus_per_task: int,
    program_args: list[str],
) -> int | None:
    """Submit one throwaway job to give the estimator its first sample.

    Stamps the same `--comment=cluv:v1:<spec_key>` as `cluv submit` so the
    record drops straight into the cache. Sets `CLUV_PROBE=1` so user scripts
    can short-circuit to a one-batch dry run instead of paying for the full job.

    After the probe lands, prints a projection at `min_samples=1` so the user
    sees what the estimator *would* suggest if subsequent runs sit near this
    one. The real estimator stays gated on the configured `min_samples`.
    """
    from salvo.history import (
        DEGENERATE_RSS_RATIO,
        JobRecord,
        estimate_mem,
        format_suggestion,
        spec_key,
    )
    from salvo.job.spec import parse_mem_mb

    git_commit = ensure_clean_git_state()

    remotes = await sync(clusters=[cluster])
    remote = remotes[0]

    key = spec_key(str(job_script), git_commit, tuple(program_args))
    sbatch_args = [
        f"--comment={history.build_comment(key)}",
        f"--mem={mem}",
        f"--time={time}",
        f"--cpus-per-task={cpus_per_task}",
    ]
    env_overrides = {
        "CLUV_SPEC_KEY": key,
        "CLUV_PROBE": "1",
        "SBATCH_MEM": mem,
    }

    console.log(
        f"probe: submitting on {cluster} with mem={mem} time={time} "
        f"cpus-per-task={cpus_per_task}"
    )
    result = await sbatch(
        remote, job_script, sbatch_args, program_args, git_commit, env_overrides
    )
    if result.returncode != 0:
        console.print(f"[red]probe sbatch failed: {result.stderr}[/red]")
        return None

    job_id = int(result.stdout.strip())
    console.log(
        f"probe job {job_id} submitted on {cluster}.\n"
        f"Use `ssh {cluster} sacct -j {job_id}` to view its status."
    )

    state = await _wait_terminal(remote, job_id)
    max_rss = await get_max_rss_mb(remote, job_id)
    elapsed_s = await get_elapsed_s(remote, job_id)
    try:
        mem_mb = parse_mem_mb(mem)
    except ValueError:
        mem_mb = 0

    history.save_record(
        JobRecord(
            job_id=str(job_id),
            key=key,
            cluster=cluster,
            state=state,
            mem_mb=mem_mb,
            max_rss_mb=max_rss,
            elapsed_s=elapsed_s,
            submitted_at=datetime.now(UTC),
        )
    )

    elapsed_str = f"{elapsed_s}s" if elapsed_s is not None else "?"
    console.log(
        f"probe job {job_id} reached {state}; "
        f"MaxRSS={max_rss}M, asked={mem_mb}M, elapsed={elapsed_str}"
    )

    asked_s = _parse_time_to_s(time)
    if asked_s and elapsed_s and elapsed_s > PROBE_WALL_FRAC * asked_s:
        console.log(
            f"[yellow]heads-up:[/yellow] probe used {elapsed_s}s of {asked_s}s asked. "
            "If your script ignores CLUV_PROBE=1, the probe costs as much as a full submit. "
            'Short-circuit with `[ -n "$CLUV_PROBE" ] && { one_batch; exit; }`.'
        )

    cfg = EstimateConfig(enabled=True)
    records = history.load(cluster, key)
    # A probe deliberately asks far more than the workload needs, so the
    # salvo degenerate-MaxRSS heuristic (which falls back to mem_mb when
    # MaxRSS<5% of the ask) would over-state the projection. Skip the
    # projection in that case and tell the user what really happened.
    degenerate = (
        max_rss is not None
        and mem_mb > 0
        and max_rss < mem_mb * DEGENERATE_RSS_RATIO
    )
    est = estimate_mem(
        records,
        safety=cfg.safety,
        window=cfg.window,
        min_samples=cfg.min_samples,
        current_ask_mb=mem_mb if mem_mb > 0 else None,
    )
    if est.mem_mb is not None:
        console.log(f"estimator: {est.rationale} (confidence={est.confidence})")
    elif degenerate:
        console.log(
            f"estimator: 1 sample cached, but MaxRSS={max_rss}M is implausibly "
            f"small for a {mem_mb}M ask (cluster's sacct sampling likely missed "
            "the peak). Submit a real run to get a usable measurement."
        )
    else:
        projection = estimate_mem(
            records,
            safety=cfg.safety,
            window=cfg.window,
            min_samples=1,
            current_ask_mb=mem_mb if mem_mb > 0 else None,
        )
        n_have = len(records)
        n_more = max(cfg.min_samples - n_have, 0)
        if projection.mem_mb is not None:
            console.log(
                f"estimator: {n_have} sample(s) cached; "
                f"needs {n_more} more real submit(s) before activating. "
                f"if future runs sit near this one, it would suggest {projection.mem_mb}M."
            )
        else:
            console.log(f"estimator: {est.rationale}")
    suggestion = format_suggestion(est)
    if suggestion is not None:
        console.log(f"suggest: {suggestion}")
    return job_id


def _parse_time_to_s(raw: str) -> int | None:
    """Parse an sbatch ``--time`` value to seconds.

    Accepts the documented forms: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``,
    ``D-HH:MM``, ``D-HH:MM:SS``. Without a leading ``D-``, two colon-parts are
    ``MM:SS`` (not ``HH:MM``); sbatch is explicit about this. Returns None on
    anything unparsable so the probe still functions; the heads-up just gets
    skipped.
    """
    raw = raw.strip()
    if not raw:
        return None
    days = 0
    if "-" in raw:
        d_str, _, rest = raw.partition("-")
        try:
            days = int(d_str)
        except ValueError:
            return None
        parts = rest.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 1:  # D-HH
            h, m, s = nums[0], 0, 0
        elif len(nums) == 2:  # D-HH:MM
            h, m, s = nums[0], nums[1], 0
        elif len(nums) == 3:  # D-HH:MM:SS
            h, m, s = nums[0], nums[1], nums[2]
        else:
            return None
    else:
        parts = raw.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 1:  # MM
            h, m, s = 0, nums[0], 0
        elif len(nums) == 2:  # MM:SS
            h, m, s = 0, nums[0], nums[1]
        elif len(nums) == 3:  # HH:MM:SS
            h, m, s = nums[0], nums[1], nums[2]
        else:
            return None
    return days * 86400 + h * 3600 + m * 60 + s
