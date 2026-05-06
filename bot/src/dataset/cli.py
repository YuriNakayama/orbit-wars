"""Typer CLI: `python -m dataset <command>`.

Top-level commands (selfplay):
  run / list / replay-inspect

Kaggle sub-app (`python -m dataset kaggle ...`):
  scrape / list / inspect
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from dataset.kaggle import scraper
from dataset.kaggle.types import ScrapeSpec
from dataset.selfplay import report
from dataset.selfplay.runner import RunSpec, run_episodes
from dataset.storage import loader
from utils.repo_root import absolute_under_repo


def _configure_logging() -> None:
    """root logger に stdout 向け INFO ハンドラを 1 度だけ設定する。

    すでに root logger に handler が付いている場合 (e.g. テスト環境) は触らない。
    """

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(min(root.level, logging.INFO))
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


app = typer.Typer(add_completion=False, help="Orbit Wars dataset CLI.")
kaggle_app = typer.Typer(add_completion=False, help="Kaggle episode scraper.")
app.add_typer(kaggle_app, name="kaggle")

console = Console()

# Anchor defaults to the repo root so `uv run --directory bot ...` does not
# silently write to `bot/data/...`. CWD-relative overrides via `--data-root`
# still work (typer keeps the user-supplied path unchanged).
DEFAULT_DATA_ROOT = absolute_under_repo("data/lake/selfplay", start=Path(__file__))
DEFAULT_KAGGLE_ROOT = absolute_under_repo(
    "data/lake/kaggle_episodes", start=Path(__file__)
)


def _parse_agents(value: str) -> tuple[str, ...]:
    parts = tuple(a.strip() for a in value.split(",") if a.strip())
    if not parts:
        raise typer.BadParameter("--agents must list at least one agent")
    return parts


def _parse_modes(value: str) -> tuple[str, ...]:
    parts = tuple(m.strip() for m in value.split(",") if m.strip())
    if not parts:
        raise typer.BadParameter("--modes must be non-empty")
    for mode in parts:
        if mode not in {"1v1", "ffa4"}:
            raise typer.BadParameter(f"unsupported mode: {mode}")
    return parts


@app.command()
def run(
    agents: str = typer.Option(..., "--agents", help="Comma-separated agent names."),
    mode: str = typer.Option("1v1", "--mode", help="'1v1' or 'ffa4'."),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Episodes to run."),
    seed: int = typer.Option(0, "--seed", help="Base seed; i-th episode uses seed+i."),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Worker processes."),
    save_replay: bool = typer.Option(
        True, "--save-replay/--no-save-replay", help="Persist env replays as json.gz."
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", help="Root directory for matches/ output."
    ),
    dvc_add: bool = typer.Option(
        False,
        "--dvc-add/--no-dvc-add",
        help="Run `dvc add <data_root>/matches` after persisting.",
    ),
) -> None:
    """Run N episodes and persist records + replays."""
    agent_names = _parse_agents(agents)
    spec = RunSpec(
        agents=agent_names,
        mode=mode,
        episodes=episodes,
        seed=seed,
        parallel=parallel,
        save_replay=save_replay,
        data_root=data_root,
    )
    with Progress(console=console) as progress:
        records = run_episodes(spec, progress=progress)
    report.summarize(records, console=console, title=f"Summary — {mode}")
    if dvc_add:
        _dvc_add_matches(data_root)


def _find_dvc_root(start: Path) -> Path | None:
    """`start` から親ディレクトリへ遡り、最初に見つかった `.dvc/` の親を返す。"""
    for candidate in (start, *start.parents):
        if (candidate / ".dvc").is_dir():
            return candidate
    return None


def _dvc_add_matches(data_root: Path) -> None:
    """Run `dvc add <matches>` from the DVC project root that owns the target.

    Worktree から外部パスを指定された場合でも、target の実体側に `.dvc/` がある
    なら、そこを CWD として `dvc add` を呼ぶ。Skips with a warning if `dvc` is
    not on PATH or the target is missing.
    """
    target = (data_root / "matches").resolve()
    if not target.exists():
        console.print(f"[yellow]--dvc-add skipped: {target} does not exist[/yellow]")
        return
    dvc_bin = shutil.which("dvc")
    if dvc_bin is None:
        console.print("[yellow]--dvc-add skipped: `dvc` not found on PATH[/yellow]")
        return
    dvc_root = _find_dvc_root(target.parent)
    if dvc_root is None:
        console.print(
            f"[red]--dvc-add failed: no `.dvc/` directory found above {target}[/red]"
        )
        raise typer.Exit(code=1)
    rel_target = target.relative_to(dvc_root)
    cmd = [dvc_bin, "add", str(rel_target)]
    console.print(f"[cyan]running (cwd={dvc_root}): {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, check=False, cwd=dvc_root)
    if result.returncode != 0:
        console.print(
            f"[red]dvc add exited with code {result.returncode}; "
            "review output above and resolve manually[/red]"
        )
        raise typer.Exit(code=result.returncode)
    console.print(
        f"[green]dvc add complete; commit `{rel_target}.dvc` and run `dvc push` "
        "when ready[/green]"
    )


@app.command("list")
def list_cmd(
    mode: str | None = typer.Option(None, "--mode", help="Filter by mode."),
    limit: int = typer.Option(20, "--limit", help="Number of rows to display."),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
) -> None:
    """List recent matches from the parquet index."""
    df = loader.list_matches(data_root=data_root, mode=mode, limit=limit)
    if df.is_empty():
        console.print("[yellow]no matches found[/yellow]")
        return
    console.print(df)


@app.command("replay-inspect")
def replay_inspect(
    match_id: str = typer.Argument(..., help="match_id to inspect."),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
) -> None:
    """Print terminal observation and winner for a stored replay."""
    payload = loader.load_replay_payload(match_id, data_root=data_root)
    steps = payload.get("steps", [])
    if not steps:
        console.print("[red]replay has no steps[/red]")
        raise typer.Exit(code=1)
    final = steps[-1]
    active = [i for i, s in enumerate(final) if s.get("status") == "ACTIVE"]
    console.print(f"turns: {len(steps)}")
    console.print(f"active players: {active}")
    for i, s in enumerate(final):
        console.print(f"  p{i}: status={s.get('status')} reward={s.get('reward')}")


def _render_scrape_summary(result: scraper.ScrapeResult) -> None:
    table = Table(title=f"Scrape summary — run_id={result.run_id}")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("teams_scanned", str(result.teams_scanned))
    table.add_row("teams_without_submission", str(result.teams_without_submission))
    table.add_row("episodes_considered", str(result.episodes_considered))
    table.add_row("skipped_existing", str(result.episodes_skipped_existing))
    table.add_row("skipped_failed", str(result.episodes_skipped_failed))
    table.add_row("skipped_mode", str(result.episodes_skipped_mode))
    table.add_row("planned", str(result.episodes_planned))
    table.add_row("fetched", str(result.episodes_fetched))
    table.add_row("failed", str(result.episodes_failed))
    table.add_row("records_written", str(result.records_written))
    table.add_row("replays_written", str(result.replays_written))
    table.add_row("dry_run", str(result.dry_run))
    console.print(table)


@kaggle_app.command("scrape")
def kaggle_scrape(
    top: int = typer.Option(20, "--top", help="Leaderboard top N teams."),
    modes: str = typer.Option("1v1,ffa4", "--modes", help="Comma-separated modes."),
    limit_per_team: int | None = typer.Option(
        None, "--limit-per-team", help="Max episodes per team (None=all)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Skip FS writes; show summary only."
    ),
    include_failed: bool = typer.Option(
        False, "--include-failed", help="Include ERROR/INVALID episodes."
    ),
    data_root: Path = typer.Option(
        DEFAULT_KAGGLE_ROOT, "--data-root", help="Root directory for output."
    ),
    dvc_add: bool = typer.Option(
        False,
        "--dvc-add/--no-dvc-add",
        help="Run `dvc add <data_root>/matches` after persisting.",
    ),
) -> None:
    """Fetch top-team episodes and persist records + replays."""

    _configure_logging()
    spec = ScrapeSpec(
        top=top,
        modes=_parse_modes(modes),
        limit_per_team=limit_per_team,
        data_root=data_root,
        dry_run=dry_run,
        include_failed=include_failed,
    )
    result = scraper.run(spec)
    _render_scrape_summary(result)
    if dvc_add and not dry_run and result.records_written > 0:
        _dvc_add_matches(data_root)


@kaggle_app.command("list")
def kaggle_list(
    mode: str | None = typer.Option(None, "--mode", help="Filter by mode."),
    limit: int = typer.Option(20, "--limit", help="Number of rows to display."),
    data_root: Path = typer.Option(
        DEFAULT_KAGGLE_ROOT, "--data-root", help="Root directory."
    ),
) -> None:
    """List recent Kaggle matches from the parquet index."""

    df = loader.list_matches(data_root=data_root, mode=mode, limit=limit)
    if df.is_empty():
        console.print("[yellow]no matches found[/yellow]")
        return
    console.print(df)


@kaggle_app.command("inspect")
def kaggle_inspect(
    episode_id: int = typer.Argument(..., help="Kaggle episode id."),
    data_root: Path = typer.Option(
        DEFAULT_KAGGLE_ROOT, "--data-root", help="Root directory."
    ),
) -> None:
    """Print turns / status / reward for a stored Kaggle episode."""

    payload = loader.load_replay_payload(f"kaggle_ep_{episode_id}", data_root=data_root)
    steps = payload.get("steps", [])
    if not steps:
        console.print("[red]replay has no steps[/red]")
        raise typer.Exit(code=1)
    final = steps[-1]
    console.print(f"episode_id: {episode_id}")
    console.print(f"turns: {len(steps)}")
    for i, s in enumerate(final):
        console.print(f"  p{i}: status={s.get('status')} reward={s.get('reward')}")


if __name__ == "__main__":
    app()
