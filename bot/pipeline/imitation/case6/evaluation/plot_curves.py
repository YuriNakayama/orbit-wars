"""Plot training curves for imitation/case5 best iter (iter6, canonical).

Reads JSON-line training logs and emits a 2x2 PNG of loss + per-head accuracy.
case1-specific diagnostic: plot title and head names assume the case1 schema
(`val_from_acc`, `val_target_acc`, `val_ships_acc`).

Usage:
    uv run python -m pipeline.imitation.case6.evaluation.plot_curves \\
        --log /tmp/train_iter6.log --out data/output/experiment/curves.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer

DEFAULT_LOG = Path("/tmp/train_iter6.log")
DEFAULT_OUT = Path("data/output/experiment/imitation_case6_learning_curves.png")
DEFAULT_LABEL = "iter6 (focal α=0.25, canonical)"
COLOR = "#1f77b4"

JSON_RE = re.compile(r"(\{.*\})\s*$")


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        m = JSON_RE.search(line)
        if not m:
            continue
        rows.append(json.loads(m.group(1)))
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


def render(log: Path, out: Path, label: str) -> tuple[Path, dict[str, float]]:
    import matplotlib.pyplot as plt

    rows = parse_log(log)
    epochs = [r["epoch"] for r in rows]

    best_idx = min(range(len(rows)), key=lambda i: rows[i]["val_total"])
    best = rows[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_loss, ax_from, ax_target, ax_ships = axes.flatten()

    train_total = [r["train_total"] for r in rows]
    val_total = [r["val_total"] for r in rows]
    ax_loss.plot(
        epochs, train_total, color=COLOR, linestyle="--", alpha=0.6, label="train"
    )
    ax_loss.plot(epochs, val_total, color=COLOR, linewidth=2, label="val")
    ax_loss.scatter(
        [best["epoch"]],
        [best["val_total"]],
        color="red",
        zorder=5,
        s=80,
        label=f"best epoch={best['epoch']}",
    )
    ax_loss.set_title("Total loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=9)

    from_acc = [r["val_from_acc"] for r in rows]
    ax_from.plot(epochs, from_acc, color=COLOR, linewidth=2, label="val_from_acc")
    ax_from.axhline(0.5, color="k", linestyle=":", alpha=0.4, label="random (0.5)")
    ax_from.scatter(
        [best["epoch"]],
        [best["val_from_acc"]],
        color="red",
        zorder=5,
        s=80,
        label=f"best={best['val_from_acc']:.3f}",
    )
    ax_from.set_title("val_from_acc (multi-label)")
    ax_from.set_xlabel("epoch")
    ax_from.set_ylabel("acc")
    ax_from.set_ylim(0.5, 1.0)
    ax_from.grid(alpha=0.3)
    ax_from.legend(fontsize=9, loc="lower right")

    target_acc = [r["val_target_acc"] for r in rows]
    ax_target.plot(epochs, target_acc, color=COLOR, linewidth=2, label="val_target_acc")
    ax_target.axhline(
        0.125, color="k", linestyle=":", alpha=0.4, label="random (0.125)"
    )
    ax_target.axhline(
        0.40, color="red", linestyle=":", alpha=0.5, label="ceiling ≈0.40"
    )
    ax_target.scatter(
        [best["epoch"]],
        [best["val_target_acc"]],
        color="red",
        zorder=5,
        s=80,
        label=f"best={best['val_target_acc']:.3f}",
    )
    ax_target.set_title("val_target_acc (8-class) — BOTTLENECK")
    ax_target.set_xlabel("epoch")
    ax_target.set_ylabel("acc")
    ax_target.set_ylim(0.10, 0.50)
    ax_target.grid(alpha=0.3)
    ax_target.legend(fontsize=9, loc="lower right")

    ships_acc = [r["val_ships_acc"] for r in rows]
    ax_ships.plot(epochs, ships_acc, color=COLOR, linewidth=2, label="val_ships_acc")
    ax_ships.axhline(0.25, color="k", linestyle=":", alpha=0.4, label="random (0.25)")
    ax_ships.scatter(
        [best["epoch"]],
        [best["val_ships_acc"]],
        color="red",
        zorder=5,
        s=80,
        label=f"best={best['val_ships_acc']:.3f}",
    )
    ax_ships.set_title("val_ships_acc (4-class)")
    ax_ships.set_xlabel("epoch")
    ax_ships.set_ylabel("acc")
    ax_ships.set_ylim(0.4, 0.9)
    ax_ships.grid(alpha=0.3)
    ax_ships.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"imitation/case5 — {label} (vs baseline_v1: 0/100)",
        fontsize=13,
    )
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out, best


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    log: Path = typer.Option(DEFAULT_LOG, "--log", "-l"),  # noqa: B008
    out: Path = typer.Option(DEFAULT_OUT, "--out", "-o"),  # noqa: B008
    label: str = typer.Option(DEFAULT_LABEL, "--label"),
) -> None:
    out_path, best = render(log, out, label)
    typer.echo(f"saved: {out_path}")
    typer.echo(
        f"best epoch={best['epoch']}  val_total={best['val_total']:.4f}  "
        f"from_acc={best['val_from_acc']:.4f}  "
        f"target_acc={best['val_target_acc']:.4f}  "
        f"ships_acc={best['val_ships_acc']:.4f}"
    )


if __name__ == "__main__":
    app()
