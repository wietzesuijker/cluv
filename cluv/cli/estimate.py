"""`cluv estimate`: dry-run the memory estimator for the current spec."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.table import Table

from cluv import history as history_module
from cluv.cli.login import get_remote_without_2fa_prompt
from cluv.cli.submit import ensure_clean_git_state
from cluv.config import get_config
from cluv.utils import console

logger = logging.getLogger(__name__)

__all__ = ["estimate"]


async def estimate(
    cluster: str,
    job_script: Path,
    program_args: list[str],
    backfill: bool,
) -> None:
    """Show what `cluv submit` would predict for memory, without submitting anything."""
    from salvo.history import estimate_mem, format_suggestion, spec_key

    git_commit = ensure_clean_git_state()
    key = spec_key(str(job_script), git_commit, tuple(program_args))

    cluv_config = get_config()
    cfg = cluv_config.estimate
    safety = cfg.safety if cfg else 1.2
    window = cfg.window if cfg else 20
    min_samples = cfg.min_samples if cfg else 3
    current_ask_mb = _current_sbatch_mem_mb(cluv_config, cluster)

    console.print(f"spec key: [bold]{key}[/bold]")
    console.print(f"cluster:  {cluster}")
    console.print(f"safety={safety}  window={window}  min_samples={min_samples}")

    records = history_module.load(cluster, key)
    if not records and backfill:
        remote = await get_remote_without_2fa_prompt(cluster)
        if remote is None:
            console.print(
                f"[yellow]no active connection to {cluster}; skipping sacct backfill.[/yellow] "
                f"Run `cluv login {cluster}` first to enable backfill."
            )
        else:
            try:
                n = await history_module.backfill_from_sacct(remote, cluster)
                console.print(f"backfilled {n} record(s) from sacct on {cluster}")
            except Exception as err:
                console.print(f"[yellow]backfill failed: {err}[/yellow]")
            records = history_module.load(cluster, key)

    if not records:
        console.print("[yellow]no records for this key; estimator would skip override.[/yellow]")
        return

    table = Table(title=f"history ({len(records)} record(s), newest first)")
    table.add_column("job id")
    table.add_column("state")
    table.add_column("mem_mb", justify="right")
    table.add_column("max_rss_mb", justify="right")
    table.add_column("submitted_at")
    for r in records[:window]:
        table.add_row(
            r.job_id,
            r.state,
            str(r.mem_mb),
            "-" if r.max_rss_mb is None else str(r.max_rss_mb),
            r.submitted_at.isoformat(timespec="minutes"),
        )
    console.print(table)

    est = estimate_mem(
        records,
        safety=safety,
        window=window,
        min_samples=min_samples,
        current_ask_mb=current_ask_mb,
    )
    console.print(f"\n[bold]estimate:[/bold] {est.rationale}")
    console.print(f"  confidence: {est.confidence}")
    console.print(f"  n_samples:  {est.n_samples}")
    if est.p95_mb is not None:
        console.print(f"  p95_mb:     {est.p95_mb}")
    if est.growth_slope_mb_per_run is not None:
        console.print(f"  growth_slope_mb_per_run: {est.growth_slope_mb_per_run:.1f}")
    if est.mem_mb is None:
        console.print("[yellow]→ SBATCH_MEM would be left untouched.[/yellow]")
    else:
        console.print(f"[green]→ SBATCH_MEM would be set to {est.mem_mb}M.[/green]")
    suggestion = format_suggestion(est)
    if suggestion is not None:
        console.print(f"[cyan]suggest:[/cyan] {suggestion}")


def _current_sbatch_mem_mb(cluv_config, cluster: str) -> int | None:
    """Resolve the configured `SBATCH_MEM` for ``cluster`` as integer MiB.

    Mirrors the merge order in ``cluv.cli.submit._initial_mem`` (global env then
    per-cluster env). Returns None when nothing is configured or the value is
    unparsable; the suggestion is silently skipped in that case.
    """
    from cluv.config import ClusterConfig
    from cluv.history import parse_mem_to_mb

    merged = {**cluv_config.env, **cluv_config.clusters.get(cluster, ClusterConfig()).env}
    raw = merged.get("SBATCH_MEM")
    if not raw:
        return None
    return parse_mem_to_mb(raw)
