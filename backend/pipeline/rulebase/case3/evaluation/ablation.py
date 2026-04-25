"""Ablation helper: toggle a case3 config flag, run N games vs case1, report v2 win rate."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False)

CONFIG_PATH = Path("pipeline/rulebase/case3/baseline/core/config.py")
DATA_ROOT = Path("data")


def _set_flag(name: str, value: bool) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = rf"^{name}:\s*bool\s*=\s*(True|False)\b"
    new_line = f"{name}: bool = {value}"
    updated, n = re.subn(pattern, new_line, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError(f"flag {name} not found in {CONFIG_PATH}")
    CONFIG_PATH.write_text(updated, encoding="utf-8")


def _run(agents: list[str], episodes: int, seed: int, parallel: int) -> Path:
    """Run episodes, return the run parquet path (newest created after call)."""
    run_dir = DATA_ROOT / "matches" / "index.parquet" / "mode=1v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in run_dir.glob("run_*.parquet")}
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "env",
        "run",
        "--agents",
        ",".join(agents),
        "--mode",
        "1v1",
        "-n",
        str(episodes),
        "--seed",
        str(seed),
        "-p",
        str(parallel),
        "--no-save-replay",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("env run failed")
    after = [p for p in run_dir.glob("run_*.parquet") if p.name not in before]
    if not after:
        raise RuntimeError("no new run parquet detected")
    return max(after, key=lambda p: p.stat().st_mtime)


def _win_rate(path: Path) -> tuple[int, int, int, int]:
    df = pl.read_parquet(path)
    v2_wins = 0
    v1_wins = 0
    draws = 0
    for row in df.iter_rows(named=True):
        if row["draw"]:
            draws += 1
            continue
        w = row["winner"]
        winner_name = row[f"agent_{w}_name"]
        if winner_name == "baseline_v3":
            v2_wins += 1
        else:
            v1_wins += 1
    return v2_wins, v1_wins, draws, len(df)


@app.command()
def ablate(
    flag: str = typer.Option(
        ..., "--flag", help="Name of bool flag to toggle to False."
    ),
    episodes_per_side: int = typer.Option(10, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    parallel: int = typer.Option(4, "-p"),
    restore_after: bool = typer.Option(True, "--restore/--no-restore"),
) -> None:
    """Disable FLAG, run games (both seat orders), report v2 win rate, restore."""
    original_value = True
    try:
        _set_flag(flag, False)
        typer.echo(f"[ablate] disabled {flag}")

        run_a = _run(["baseline_v3", "baseline_v1"], episodes_per_side, seed, parallel)
        run_b = _run(
            ["baseline_v1", "baseline_v3"], episodes_per_side, seed + 500, parallel
        )

        wa, la, da, na = _win_rate(run_a)
        wb, lb, db, nb = _win_rate(run_b)
        total_w = wa + wb
        total_d = da + db
        total_n = na + nb
        typer.echo(
            f"[ablate] flag={flag} v2 wins {total_w}/{total_n} "
            f"(seat0={wa}/{na}, seat1={wb}/{nb}, "
            f"losses={la + lb}, draws={total_d})"
        )
    finally:
        if restore_after:
            _set_flag(flag, original_value)
            typer.echo(f"[ablate] restored {flag}={original_value}")


@app.command()
def baseline_run(
    episodes_per_side: int = typer.Option(10, "-n"),
    seed: int = typer.Option(2000, "--seed"),
    parallel: int = typer.Option(4, "-p"),
) -> None:
    """Run v2 (current config) vs v1 both directions and report."""
    run_a = _run(["baseline_v3", "baseline_v1"], episodes_per_side, seed, parallel)
    run_b = _run(
        ["baseline_v1", "baseline_v3"], episodes_per_side, seed + 500, parallel
    )
    wa, la, da, na = _win_rate(run_a)
    wb, lb, db, nb = _win_rate(run_b)
    typer.echo(
        f"[baseline] v2 wins {wa + wb}/{na + nb} "
        f"(seat0={wa}/{na}, seat1={wb}/{nb}, draws={da + db})"
    )


if __name__ == "__main__":
    app()
