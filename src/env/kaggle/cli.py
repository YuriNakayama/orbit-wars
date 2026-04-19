"""Typer CLI: `python -m env.kaggle scrape|list|inspect`.

Kaggle 上位 team のエピソード replay を収集し、`data_root/matches/` 配下に
保存する。`scrape` は rate limit（60 req/60s）を厳守し、既取得 id を skip。
`list` / `inspect` は収集結果の確認用。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from env import loader
from env.kaggle import scraper
from env.kaggle.types import ScrapeSpec

app = typer.Typer(add_completion=False, help="Kaggle Orbit Wars log collector.")
console = Console()

DEFAULT_DATA_ROOT = Path("data/kaggle_episodes")


def _parse_modes(value: str) -> tuple[str, ...]:
    parts = tuple(m.strip() for m in value.split(",") if m.strip())
    if not parts:
        raise typer.BadParameter("--modes must be non-empty")
    for mode in parts:
        if mode not in {"1v1", "ffa4"}:
            raise typer.BadParameter(f"unsupported mode: {mode}")
    return parts


def _render_summary(result: scraper.ScrapeResult) -> None:
    table = Table(title=f"Scrape summary — run_id={result.run_id}")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("teams_scanned", str(result.teams_scanned))
    table.add_row("teams_without_submission", str(result.teams_without_submission))
    table.add_row("episodes_considered", str(result.episodes_considered))
    table.add_row("skipped_existing", str(result.episodes_skipped_existing))
    table.add_row("skipped_failed", str(result.episodes_skipped_failed))
    table.add_row("skipped_mode", str(result.episodes_skipped_mode))
    table.add_row("fetched", str(result.episodes_fetched))
    table.add_row("failed", str(result.episodes_failed))
    table.add_row("records_written", str(result.records_written))
    table.add_row("replays_written", str(result.replays_written))
    table.add_row("dry_run", str(result.dry_run))
    console.print(table)


@app.command()
def scrape(
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
        DEFAULT_DATA_ROOT, "--data-root", help="Root directory for output."
    ),
) -> None:
    """Fetch top-team episodes and persist records + replays."""

    spec = ScrapeSpec(
        top=top,
        modes=_parse_modes(modes),
        limit_per_team=limit_per_team,
        data_root=data_root,
        dry_run=dry_run,
        include_failed=include_failed,
    )
    result = scraper.run(spec)
    _render_summary(result)


@app.command("list")
def list_cmd(
    mode: str | None = typer.Option(None, "--mode", help="Filter by mode."),
    limit: int = typer.Option(20, "--limit", help="Number of rows to display."),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", help="Root directory."
    ),
) -> None:
    """List recent Kaggle matches from the parquet index."""

    df = loader.list_matches(data_root=data_root, mode=mode, limit=limit)
    if df.is_empty():
        console.print("[yellow]no matches found[/yellow]")
        return
    console.print(df)


@app.command()
def inspect(
    episode_id: int = typer.Argument(..., help="Kaggle episode id."),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", help="Root directory."
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
