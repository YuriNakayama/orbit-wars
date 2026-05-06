"""Phase 1-2: temperature scaling for target_head / ships_head on val.parquet.

For each head we minimize val NLL over T ∈ [0.5, 3.0], measure ECE, and save
the optimal T* to data/output/experiment/imitation_case1_calibration.json.

Only the fired-source rows are used (consistent with how training-time losses
compute target/ships accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pipeline.imitation.case6.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case6.policy.types import BatchFeatures
from pipeline.imitation.case6.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)

CONFIG_PATH = Path("pipeline/imitation/case5/configs/il_baseline.yaml")
WEIGHTS_PATH = Path("pipeline/imitation/case5/policy/weights.pt")
REPORT_PATH = Path("data/output/experiment/imitation_case1_calibration.json")
SHIPS_BUCKETS = 4


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
        ships_buckets=int(mcfg.get("ships_buckets", SHIPS_BUCKETS)),
    )
    model = DeepSetsPolicy(model_cfg)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _collect_fired(
    model: DeepSetsPolicy, val_ds: CaseThreeDataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
    target_logits_all: list[np.ndarray] = []
    target_gt_all: list[np.ndarray] = []
    ships_logits_all: list[np.ndarray] = []
    ships_gt_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(_to_batch_features(batch))
            fired = batch.from_multihot & batch.my_planet_mask
            if not fired.any():
                continue
            b_idx, s_idx = fired.nonzero(as_tuple=True)
            target_logits_all.append(output.target_logits[b_idx, s_idx].cpu().numpy())
            target_gt_all.append(batch.target_per_src[b_idx, s_idx].cpu().numpy())
            ships_logits_all.append(output.ships_logits[b_idx, s_idx].cpu().numpy())
            ships_gt_all.append(batch.ships_per_src[b_idx, s_idx].cpu().numpy())
    return (
        np.concatenate(target_logits_all),
        np.concatenate(target_gt_all),
        np.concatenate(ships_logits_all),
        np.concatenate(ships_gt_all),
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)  # type: ignore[no-any-return]


def _nll(logits: np.ndarray, gt: np.ndarray, T: float) -> float:
    scaled = logits / max(T, 1e-6)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    logZ = np.log(np.exp(scaled).sum(axis=-1))
    logp_true = scaled[np.arange(len(gt)), gt] - logZ
    return float(-logp_true.mean())


def _ece(probs: np.ndarray, gt: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == gt).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(gt)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        avg_conf = conf[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def _sweep(logits: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    Ts = [round(0.5 + 0.1 * i, 2) for i in range(26)]  # 0.5..3.0
    rows = []
    for T in Ts:
        nll = _nll(logits, gt, T)
        probs = _softmax(logits / T)
        ece = _ece(probs, gt)
        rows.append({"T": T, "nll": nll, "ece": ece})
    best_nll = min(rows, key=lambda r: r["nll"])
    best_ece = min(rows, key=lambda r: r["ece"])
    base_nll = _nll(logits, gt, 1.0)
    base_ece = _ece(_softmax(logits), gt)
    return {
        "baseline_T1": {"nll": base_nll, "ece": base_ece},
        "best_by_nll": best_nll,
        "best_by_ece": best_ece,
        "sweep": rows,
    }


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    val_path = Path(cfg["data"]["out_val"])
    val_ds = CaseThreeDataset(val_path)
    model = _load_model()
    t_logits, t_gt, s_logits, s_gt = _collect_fired(model, val_ds)

    report = {
        "weights": str(WEIGHTS_PATH),
        "val_parquet": str(val_path),
        "n_fired": int(len(t_gt)),
        "target_head": _sweep(t_logits, t_gt),
        "ships_head": _sweep(s_logits, s_gt),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"saved: {REPORT_PATH}")

    t_best = report["target_head"]["best_by_nll"]  # type: ignore[index]
    t_base = report["target_head"]["baseline_T1"]  # type: ignore[index]
    s_best = report["ships_head"]["best_by_nll"]  # type: ignore[index]
    s_base = report["ships_head"]["baseline_T1"]  # type: ignore[index]
    print(
        f"target: T*={t_best['T']}  NLL {t_base['nll']:.4f} → {t_best['nll']:.4f}  "
        f"ECE {t_base['ece']:.4f} → {t_best['ece']:.4f}"
    )
    print(
        f"ships:  T*={s_best['T']}  NLL {s_base['nll']:.4f} → {s_best['nll']:.4f}  "
        f"ECE {s_base['ece']:.4f} → {s_best['ece']:.4f}"
    )


if __name__ == "__main__":
    main()
