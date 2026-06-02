"""Visualize reinforce/case5 support_reward training curves across iters.

Reads each run's metrics.json (history list of per-iter dicts) and plots a
4-panel comparison: win_rate (lite phase), value_loss, entropy, approx_kl.
Saves a PNG under data/output/experiment/reinforce/case5/.

Dev-only EDA script (excluded from the Kaggle submission via .submitignore).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_MODELS = _ROOT / "data/output/models/reinforce"

RUNS: list[tuple[str, str]] = [
    (
        "H1 combined (coef 0.001/0.50)",
        "case5_kaggle_jax_train/runs/"
        "20260527-151636__feature-support-reward__2f37b9e__seed0",
    ),
    (
        "H2 ratio (coef 0.50)",
        "case5_kaggle_jax_train_h2_ratio/runs/"
        "20260527-182312__feature-support-reward__c359b68__seed0",
    ),
    (
        "H4 ratio coef=1.0 (best)",
        "case5_kaggle_jax_train_h4_ratio_coef1/runs/"
        "20260528-004854__feature-support-reward__8e2e4a3__seed0",
    ),
    (
        "H5 ratio_prod (rejected)",
        "case5_kaggle_jax_train_h5_ratio_prod/runs/"
        "20260528-050116__feature-support-reward__531f725__seed0",
    ),
]


def _smooth(y: list[float], k: int = 10) -> np.ndarray:
    """Trailing moving average of window k."""
    arr = np.asarray(y, dtype=float)
    out = np.copy(arr)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - k + 1) : i + 1].mean()
    return out


def _load(rel: str) -> list[dict[str, float]]:
    path = _MODELS / rel / "metrics.json"
    return list(json.loads(path.read_text())["history"])


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        ("win_rate", "Win rate (smoothed, 10-iter MA)", axes[0][0]),
        ("value_loss", "Value loss", axes[0][1]),
        ("entropy", "Policy entropy", axes[1][0]),
        ("approx_kl", "Approx KL", axes[1][1]),
    ]
    for label, rel in RUNS:
        hist = _load(rel)
        x = np.arange(len(hist))
        for key, _title, ax in panels:
            y = [float(r.get(key, 0.0)) for r in hist]
            plotted = _smooth(y) if key == "win_rate" else np.asarray(y, dtype=float)
            ax.plot(x, plotted, label=label, linewidth=1.6)

    for key, title, ax in panels:
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if key == "win_rate":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="_baseline 0.50")
            ax.set_ylim(0.0, 1.0)
        if key == "approx_kl":
            ax.axhline(0.1, color="red", ls="--", lw=0.8)

    labels = " / ".join(label for label, _ in RUNS)
    fig.suptitle(
        f"reinforce/case5 support_reward — training curves ({labels})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = _ROOT / "data/output/experiment/reinforce/case5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "training_curves_all.png"
    fig.savefig(out, dpi=120)
    print(str(out))


if __name__ == "__main__":
    main()
