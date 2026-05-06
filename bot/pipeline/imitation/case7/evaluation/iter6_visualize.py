"""iter6: training loss curve + grouped permutation importance を可視化。

Outputs:
- data/output/experiment/imitation_case7_iter6_grouped_importance.json
- data/output/experiment/imitation_case7_iter6_analysis.png

Run:
    uv run --directory bot python -m pipeline.imitation.case7.evaluation.iter6_visualize
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from pipeline.imitation.case7.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case7.policy.types import BatchFeatures
from pipeline.imitation.case7.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
BOT_ROOT = REPO_ROOT / "bot"
CONFIG = BOT_ROOT / "pipeline/imitation/case7/configs/il_case7.yaml"
WEIGHTS = BOT_ROOT / "pipeline/imitation/case7/policy/weights.pt"
RUN_DIR_BASE = REPO_ROOT / "data/output/models/imitation/case7/runs"
RUN_DIR = RUN_DIR_BASE / (
    "20260506-010029__feature-feature-engineering__0895b65__seed0"
)
OUT_DIR = REPO_ROOT / "data/output/experiment"

# (label, run_id, dim_str, status_color)
ITER_RUNS: list[tuple[str, str, str, str]] = [
    (
        "iter1 (24/10)",
        "20260505-040940__feature-feature-engineering__5a89082__seed0",
        "PLANET=24, GLOBAL=10",
        "#7f7f7f",
    ),
    (
        "iter2 (34/14)",
        "20260505-062823__feature-feature-engineering__850c34e__seed0",
        "PLANET=34, GLOBAL=14",
        "#1f77b4",
    ),
    (
        "iter3 (63/14)",
        "20260505-080208__feature-feature-engineering__7219177__seed0",
        "PLANET=63, GLOBAL=14",
        "#2ca02c",
    ),
    (
        "iter4 (61/12) LEAK",
        "20260505-102543__feature-feature-engineering__8fe82ae__seed0",
        "PLANET=61, GLOBAL=12",
        "#d62728",
    ),
    (
        "iter6 (57/8) clean",
        "20260506-010029__feature-feature-engineering__0895b65__seed0",
        "PLANET=57, GLOBAL=8",
        "#9467bd",
    ),
]
SHIPS_BUCKETS = 4

# iter6 layout (PLANET=57, GLOBAL=8) — iter5 と group 定義は同じ、K2/K4 を除外し
# K3 frontline_distance を planet 53-56 に shift。
PLANET_GROUPS: dict[str, list[int]] = {
    "G1_position_geometry": [0, 1, 2, 3, 4, 5, 6],
    "G2_ships_production_owner": [7, 8, 9, 10, 11, 12],
    "G3_incoming_eta": [13, 14],
    "G4_ship_prediction_h30": [15, 16, 17, 18, 19, 20],
    "G5_history_obs_n2_n3": [19, 20, 21],
    # iter1 ship-event は iter4 で削除済 (cols 22-25 に G7 が入る)
    "G7_inbound_fleet_trajectory": [22, 23, 24, 25],
    "G8_multi_horizon": [26, 27, 28, 29],
    "iter2_home_centroid_per_planet": [30, 31],
    "G9_pairwise_top_k_k3": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    "G10_defense_surplus": [44, 45, 46, 47],
    "G11_sparse_mask": [48, 49, 50, 51, 52],
    "K3_frontline_distance": [53, 54, 55, 56],
}

GLOBAL_GROUPS: dict[str, list[int]] = {
    "H2_global_aggregate_resources": [0, 1, 2, 3],
    "H3_global_launch_history": [4, 5, 6, 7],
}


@dataclass
class HeadMetrics:
    from_f1: float
    target_f1: float
    ships_f1: float


class GroupDelta(TypedDict):
    n_col: int
    delta_from_f1: float
    delta_target_f1: float
    delta_ships_f1: float
    sum_abs_delta: float


class BaseMetrics(TypedDict):
    from_f1: float
    target_f1: float
    ships_f1: float


class ImportanceReport(TypedDict):
    base: BaseMetrics
    groups: dict[str, GroupDelta]


def _to_batch_features(sample: BatchedSample) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats,
        planet_mask=sample.planet_mask,
        my_planet_mask=sample.my_planet_mask,
        target_mask=sample.target_mask,
        global_feats=sample.global_feats,
        template_ctx=sample.template_ctx,
    )


def _load_model() -> DeepSetsPolicy:
    cfg_yaml = yaml.safe_load(CONFIG.read_text())
    mcfg = cfg_yaml["model"]
    model_cfg = ModelConfig(
        planet_in_dim=int(mcfg["planet_in_dim"]),
        global_in_dim=int(mcfg["global_in_dim"]),
        hidden=int(mcfg.get("hidden", 128)),
        ships_buckets=int(mcfg.get("ships_buckets", SHIPS_BUCKETS)),
    )
    model = DeepSetsPolicy(model_cfg)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate(
    model: DeepSetsPolicy,
    val_ds: CaseThreeDataset,
    perturb: tuple[str, list[int]] | None = None,
    rng: np.random.Generator | None = None,
) -> HeadMetrics:
    """val 全体で 3 head F1。perturb=(scope, cols) が指定されたら shuffle。"""
    loader: DataLoader[BatchedSample] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=256,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    rng = rng or np.random.default_rng(0)

    from_logits_all: list[np.ndarray] = []
    from_gt_all: list[np.ndarray] = []
    my_mask_all: list[np.ndarray] = []
    target_pred_all: list[np.ndarray] = []
    target_gt_all: list[np.ndarray] = []
    ships_pred_all: list[np.ndarray] = []
    ships_gt_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features = _to_batch_features(batch)
            if perturb is not None:
                scope, cols = perturb
                if scope == "planet":
                    pf = features.planet_feats.clone()
                    B, P, _ = pf.shape
                    flat = pf.reshape(B * P, -1)
                    perm = rng.permutation(B * P)
                    flat[:, cols] = flat[perm][:, cols]
                    features = BatchFeatures(
                        planet_feats=flat.reshape(B, P, -1),
                        planet_mask=features.planet_mask,
                        my_planet_mask=features.my_planet_mask,
                        target_mask=features.target_mask,
                        global_feats=features.global_feats,
                        template_ctx=features.template_ctx,
                    )
                elif scope == "global":
                    gf = features.global_feats.clone()
                    B = gf.shape[0]
                    perm = rng.permutation(B)
                    gf[:, cols] = gf[perm][:, cols]
                    features = BatchFeatures(
                        planet_feats=features.planet_feats,
                        planet_mask=features.planet_mask,
                        my_planet_mask=features.my_planet_mask,
                        target_mask=features.target_mask,
                        global_feats=gf,
                        template_ctx=features.template_ctx,
                    )
            output = model(features)

            from_logits_all.append(output.from_logits.cpu().numpy())
            from_gt_all.append(batch.from_multihot.cpu().numpy())
            my_mask_all.append(batch.my_planet_mask.cpu().numpy())

            fired = batch.from_multihot & batch.my_planet_mask
            if fired.any():
                b_idx, s_idx = fired.nonzero(as_tuple=True)
                target_pred_all.append(
                    output.target_logits[b_idx, s_idx].argmax(dim=-1).cpu().numpy()
                )
                target_gt_all.append(batch.target_per_src[b_idx, s_idx].cpu().numpy())
                ships_pred_all.append(
                    output.ships_logits[b_idx, s_idx].argmax(dim=-1).cpu().numpy()
                )
                ships_gt_all.append(batch.ships_per_src[b_idx, s_idx].cpu().numpy())

    from_logits = np.concatenate(from_logits_all)
    from_gt = np.concatenate(from_gt_all).astype(bool)
    my_mask = np.concatenate(my_mask_all).astype(bool)
    pred_v = (1.0 / (1.0 + np.exp(-from_logits[my_mask])) > 0.5).astype(int)
    from_f1 = float(f1_score(from_gt[my_mask].astype(int), pred_v, zero_division=0))

    target_pred = np.concatenate(target_pred_all)
    target_gt = np.concatenate(target_gt_all)
    target_f1 = float(
        f1_score(target_gt, target_pred, average="macro", zero_division=0)
    )

    ships_pred = np.concatenate(ships_pred_all)
    ships_gt = np.concatenate(ships_gt_all)
    ships_f1 = float(f1_score(ships_gt, ships_pred, average="macro", zero_division=0))

    return HeadMetrics(from_f1=from_f1, target_f1=target_f1, ships_f1=ships_f1)


def compute_grouped_importance() -> ImportanceReport:
    cfg_yaml = yaml.safe_load(CONFIG.read_text())
    val_parquet_rel = Path(cfg_yaml["data"]["out_val"])
    val_parquet = (
        val_parquet_rel
        if val_parquet_rel.is_absolute()
        else REPO_ROOT / val_parquet_rel
    )
    val_ds = CaseThreeDataset(val_parquet)
    model = _load_model()

    base = _evaluate(model, val_ds)
    print(
        f"[base] from_f1={base.from_f1:.4f} target_f1={base.target_f1:.4f} "
        f"ships_f1={base.ships_f1:.4f}"
    )

    results: dict[str, GroupDelta] = {}
    rng = np.random.default_rng(0)

    for group, cols in PLANET_GROUPS.items():
        m = _evaluate(model, val_ds, perturb=("planet", cols), rng=rng)
        d_from = base.from_f1 - m.from_f1
        d_target = base.target_f1 - m.target_f1
        d_ships = base.ships_f1 - m.ships_f1
        sum_abs = abs(d_from) + abs(d_target) + abs(d_ships)
        results[group] = {
            "n_col": len(cols),
            "delta_from_f1": d_from,
            "delta_target_f1": d_target,
            "delta_ships_f1": d_ships,
            "sum_abs_delta": sum_abs,
        }
        print(
            f"[planet] {group:40s} n={len(cols):2d}  "
            f"Δfrom={d_from:+.4f}  Δtgt={d_target:+.4f}  "
            f"Δshp={d_ships:+.4f}  sum|Δ|={sum_abs:.4f}"
        )

    for group, cols in GLOBAL_GROUPS.items():
        m = _evaluate(model, val_ds, perturb=("global", cols), rng=rng)
        d_from = base.from_f1 - m.from_f1
        d_target = base.target_f1 - m.target_f1
        d_ships = base.ships_f1 - m.ships_f1
        sum_abs = abs(d_from) + abs(d_target) + abs(d_ships)
        results[group] = {
            "n_col": len(cols),
            "delta_from_f1": d_from,
            "delta_target_f1": d_target,
            "delta_ships_f1": d_ships,
            "sum_abs_delta": sum_abs,
        }
        print(
            f"[global] {group:40s} n={len(cols):2d}  "
            f"Δfrom={d_from:+.4f}  Δtgt={d_target:+.4f}  "
            f"Δshp={d_ships:+.4f}  sum|Δ|={sum_abs:.4f}"
        )

    return {
        "base": {
            "from_f1": base.from_f1,
            "target_f1": base.target_f1,
            "ships_f1": base.ships_f1,
        },
        "groups": results,
    }


def render_panel(importance: ImportanceReport) -> Path:
    metrics = json.loads((RUN_DIR / "metrics.json").read_text())
    train_loss = metrics["train_loss_history"]
    val_loss = metrics["val_loss_history"]
    best_epoch = metrics["best_epoch"]
    best_val = metrics["best_val_loss"]

    base = importance["base"]
    groups = importance["groups"]
    sorted_items = sorted(
        groups.items(), key=lambda kv: kv[1]["sum_abs_delta"], reverse=True
    )
    names = [k for k, _ in sorted_items]
    sum_abs = [v["sum_abs_delta"] for _, v in sorted_items]
    d_from = [v["delta_from_f1"] for _, v in sorted_items]
    d_target = [v["delta_target_f1"] for _, v in sorted_items]
    d_ships = [v["delta_ships_f1"] for _, v in sorted_items]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.4, wspace=0.3)

    # Panel A: training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = np.arange(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, "o-", label="train", color="#1f77b4")
    ax1.plot(epochs, val_loss, "s-", label="val", color="#d62728")
    ax1.axvline(
        best_epoch + 1,
        color="green",
        linestyle="--",
        alpha=0.6,
        label=f"best epoch={best_epoch + 1}",
    )
    ax1.axhline(
        best_val,
        color="green",
        linestyle=":",
        alpha=0.4,
        label=f"best val={best_val:.4f}",
    )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(
        "iter6 training loss (PLANET=57, GLOBAL=8, K2/K4 removed)\n"
        f"train→{train_loss[-1]:.4f}, val best={best_val:.4f} @ ep{best_epoch + 1}"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B: head F1 baseline summary
    ax2 = fig.add_subplot(gs[0, 1])
    head_names = ["from_F1", "target_macro_F1", "ships_macro_F1"]
    iter1 = [0.6651, 0.3210, 0.6425]
    iter4 = [0.6838, 0.3296, 0.9349]
    iter6 = [base["from_f1"], base["target_f1"], base["ships_f1"]]
    x = np.arange(len(head_names))
    w = 0.27
    ax2.bar(x - w, iter1, w, label="iter1 base", color="#888")
    ax2.bar(x, iter4, w, label="iter4 (LEAK)", color="#d62728")
    ax2.bar(x + w, iter6, w, label="iter6 clean", color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_names, fontsize=9)
    ax2.set_ylabel("F1")
    ax2.set_title("3 head F1 — iter1 vs iter4 (leak) vs iter6 (clean)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(iter6):
        ax2.text(i + w, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    # Panel C: grouped permutation importance (sum |Δ| ranking)
    ax3 = fig.add_subplot(gs[1, :])
    y = np.arange(len(names))
    ax3.barh(y, sum_abs, color="#1f77b4", alpha=0.4, label="sum |Δ|")
    ax3.barh(y - 0.22, np.abs(d_from), 0.22, color="#1f77b4", label="|Δfrom_F1|")
    ax3.barh(y, np.abs(d_target), 0.22, color="#ff7f0e", label="|Δtarget_F1|")
    ax3.barh(y + 0.22, np.abs(d_ships), 0.22, color="#2ca02c", label="|Δships_F1|")
    ax3.set_yticks(y)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel("|Δ F1| (vs baseline)")
    ax3.set_title(
        "iter6 grouped permutation importance - large |delta| = important group "
        "(K2/K4 removed, leak-free)"
    )
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(axis="x", alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "imitation_case7_iter6_analysis.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def render_loss_comparison() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    summary_rows = []
    for label, run_id, dim_str, color in ITER_RUNS:
        m_path = RUN_DIR_BASE / run_id / "metrics.json"
        if not m_path.exists():
            print(f"[skip] {label}: missing {m_path}")
            continue
        m = json.loads(m_path.read_text())
        train = m["train_loss_history"]
        val = m["val_loss_history"]
        best_ep = m["best_epoch"]
        best_val = m["best_val_loss"]
        epochs = np.arange(1, len(train) + 1)

        axes[0].plot(epochs, train, "-", color=color, alpha=0.8, label=label)
        axes[1].plot(epochs, val, "-", color=color, alpha=0.9, label=label, linewidth=2)
        axes[1].plot(
            best_ep + 1, best_val, "o", color=color, markersize=10, markeredgecolor="k"
        )

        summary_rows.append((label, dim_str, best_val, best_ep + 1))

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("train loss (lower is better)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val loss")
    axes[1].set_title("val loss (best marked with circle)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    iter1_best = next(r[2] for r in summary_rows if r[0].startswith("iter1"))
    txt_lines = ["iter           dim              best_val   epoch   delta_vs_iter1"]
    for label, dim_str, best_val, best_ep in summary_rows:
        delta = best_val - iter1_best
        txt_lines.append(
            f"{label:24s} {dim_str:22s} {best_val:.4f}   {best_ep:5d}   {delta:+.4f}"
        )
    fig.suptitle(
        "case7 iter1 -> iter6 training comparison\n" + "\n".join(txt_lines),
        fontsize=10,
        family="monospace",
        y=1.02,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "imitation_case7_iter_loss_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    import sys

    skip_importance = "--skip-importance" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not skip_importance:
        importance = compute_grouped_importance()
        json_out = OUT_DIR / "imitation_case7_iter6_grouped_importance.json"
        json_out.write_text(json.dumps(importance, indent=2))
        print(f"\nsaved: {json_out}")
        png_out = render_panel(importance)
        print(f"saved: {png_out}")
    loss_out = render_loss_comparison()
    print(f"saved: {loss_out}")


if __name__ == "__main__":
    main()
