"""Phase 1-1: sweep from_head threshold on val.parquet and pick F1-optimal τ*.

Loads the current canonical weights, runs the model on val data, and searches
threshold values for sigmoid(from_logits) to find the value that maximizes
positive-class F1 (over my_planet_mask valid slots only).

Result is saved to docs/experiment/imitation_case1_threshold_sweep.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from pipeline.imitation.case1.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case1.policy.types import BatchFeatures
from pipeline.imitation.case1.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)

CONFIG_PATH = Path("pipeline/imitation/case1/configs/il_baseline.yaml")
WEIGHTS_PATH = Path("pipeline/imitation/case1/policy/weights.pt")
REPORT_PATH = Path("docs/experiment/imitation_case1_threshold_sweep.json")


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
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mcfg = cfg.get("model", {})
    model_cfg = ModelConfig(
        planet_in_dim=int(mcfg.get("planet_in_dim", 11)),
        global_in_dim=int(mcfg.get("global_in_dim", 6)),
        hidden=int(mcfg.get("hidden", 128)),
        ships_buckets=int(mcfg.get("ships_buckets", 4)),
    )
    model = DeepSetsPolicy(model_cfg)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _collect(
    model: DeepSetsPolicy, val_ds: CaseThreeDataset
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
    logits_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(_to_batch_features(batch))
            logits_all.append(output.from_logits.cpu().numpy())
            gt_all.append(batch.from_multihot.cpu().numpy())
            mask_all.append(batch.my_planet_mask.cpu().numpy())
    logits = np.concatenate(logits_all)
    gt = np.concatenate(gt_all).astype(bool)
    mask = np.concatenate(mask_all).astype(bool)
    probs_v = 1.0 / (1.0 + np.exp(-logits[mask]))
    gt_v = gt[mask].astype(int)
    return probs_v, gt_v


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    val_path = Path(cfg["data"]["out_val"])
    val_ds = CaseThreeDataset(val_path)
    model = _load_model()
    probs, gt = _collect(model, val_ds)

    sweep = []
    thresholds = [round(0.01 + 0.01 * i, 3) for i in range(60)]  # 0.01..0.60
    for th in thresholds:
        pred = (probs > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        sweep.append(
            {
                "threshold": th,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "pred_pos_rate": float(pred.mean()),
            }
        )

    best = max(sweep, key=lambda r: r["f1"])
    gt_pos_rate = float(gt.mean())

    out = {
        "weights": str(WEIGHTS_PATH),
        "val_parquet": str(val_path),
        "n_valid_slots": int(len(gt)),
        "gt_positive_rate": gt_pos_rate,
        "best": best,
        "sweep": sweep,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(out, indent=2))
    print(f"saved: {REPORT_PATH}")
    print(
        f"best threshold τ*={best['threshold']}  F1={best['f1']:.4f}  "
        f"precision={best['precision']:.4f}  recall={best['recall']:.4f}  "
        f"pred_pos_rate={best['pred_pos_rate']:.4f}  (GT pos rate={gt_pos_rate:.4f})"
    )


if __name__ == "__main__":
    main()
