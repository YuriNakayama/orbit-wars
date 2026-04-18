"""Typer CLI: python -m env run|list|replay-inspect."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.progress import Progress

from env import loader, report
from env.runner import RunSpec, run_episodes

app = typer.Typer(add_completion=False, help="Orbit Wars evaluation CLI.")
console = Console()

DEFAULT_DATA_ROOT = Path("data")


def _parse_agents(value: str) -> tuple[str, ...]:
    parts = tuple(a.strip() for a in value.split(",") if a.strip())
    if not parts:
        raise typer.BadParameter("--agents must list at least one agent")
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


def _dump_parquet_head(data_root: Path, mode: str | None, limit: int) -> pl.DataFrame:
    return loader.list_matches(data_root=data_root, mode=mode, limit=limit)


if __name__ == "__main__":
    app()
