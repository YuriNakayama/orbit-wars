"""Plot a case8 training run's metrics.json as learning curves (requirement 3).

Reads the per-iter `metrics.json` (flushed every iter by train_jax, so this
works mid-training too) and renders win_rate / mean_reward / losses / entropy /
KL / opponent into a single PNG. Pairs with `dev/runpod pull` (or `sync --pull`)
to visualize an in-flight run locally.

Usage (from bot/):
    uv run python -m pipeline.reinforce.case8.evaluation.plot_metrics \
        --metrics data/output/models/reinforce/case8_train_jax/runs/<run>/metrics.json \
        --out /tmp/case8_curves.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

Row = dict[str, float | int | str | None]

_METRICS_OPT = typer.Option(..., "--metrics", help="path to metrics.json")
_OUT_OPT = typer.Option(Path("/tmp/case8_curves.png"), "--out", help="output PNG path")


def _num(value: float | int | str | None) -> float:
    return float(value)  # type: ignore[arg-type]


def _series(history: list[Row], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in history:
        if key in row and row[key] is not None:
            xs.append(int(_num(row["iter"])))
            ys.append(_num(row[key]))
    return xs, ys


def _plot_panel(ax: Axes, history: list[Row], keys: list[str], title: str) -> None:
    for key in keys:
        xs, ys = _series(history, key)
        if xs:
            ax.plot(xs, ys, marker=".", label=key)
    ax.set_title(title)
    ax.set_xlabel("iter")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _plot_win_by_opponent(
    ax: Axes, history: list[Row], key: str, title: str, *, clamp01: bool = False
) -> None:
    """Plot a per-iter metric colored/segmented by the opponent of that iter.

    Each iter's point is colored by its `opponent`, and a dashed horizontal line
    marks each opponent's mean — so the effect of a curriculum/pool switch (e.g.
    noop → baseline_jax_full → self_snapshot) is visible at a glance.
    """
    by_opp: dict[str, tuple[list[int], list[float]]] = {}
    for row in history:
        opp = str(row.get("opponent", "?"))
        if key in row and row[key] is not None:
            xs, ys = by_opp.setdefault(opp, ([], []))
            xs.append(int(_num(row["iter"])))
            ys.append(_num(row[key]))
    for opp, (xs, ys) in sorted(by_opp.items()):
        ax.scatter(xs, ys, label=opp, s=40)
        mean = sum(ys) / len(ys)
        ax.axhline(mean, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("iter")
    if clamp01:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


@app.command()
def main(metrics: Path = _METRICS_OPT, out: Path = _OUT_OPT) -> None:
    """Render learning curves from a (possibly in-flight) metrics.json."""
    data = json.loads(metrics.read_text())
    history: list[Row] = data.get("history", [])
    if not history:
        raise typer.BadParameter(f"no history rows in {metrics}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    # Row 0 — PROGRESS (matchmaking-free, the real signal under f_var):
    _plot_panel(
        axes[0, 0], history, ["heldout_win"], "★ held-out win vs fixed opponent"
    )
    axes[0, 0].axhline(0.5, ls="--", c="gray", alpha=0.6)
    axes[0, 0].set_ylim(-0.02, 1.02)
    _plot_panel(axes[0, 1], history, ["agent_elo"], "★ agent Elo (vs fixed ref)")
    _plot_win_by_opponent(
        axes[0, 2], history, "win_rate", "match win rate (≈0.5 by design)", clamp01=True
    )
    # Row 1 — HEALTH (collapse / divergence detectors):
    _plot_panel(axes[1, 0], history, ["entropy"], "policy entropy (collapse if ↓<10)")
    _plot_panel(axes[1, 1], history, ["value_loss"], "value loss")
    _plot_panel(axes[1, 2], history, ["approx_kl"], "approx KL")
    # Row 2 — context:
    _plot_win_by_opponent(axes[2, 0], history, "mean_reward", "reward by opponent")
    _plot_panel(axes[2, 1], history, ["policy_loss"], "policy loss")
    _plot_panel(
        axes[2, 2], history, ["rollout_secs", "update_secs"], "per-iter time (s)"
    )

    iters_run = int(data.get("iterations_run", len(history)))
    best = float(data.get("best_win_rate", float("nan")))
    opponents = sorted({str(r.get("opponent")) for r in history if r.get("opponent")})
    fig.suptitle(
        f"case8 run — iters={iters_run} best_win={best:.3f} opponents={opponents}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    logger.info(
        json.dumps({"event": "plot_saved", "out": str(out), "iters": iters_run})
    )
    typer.echo(f"saved {out} ({iters_run} iters, best_win={best:.3f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
