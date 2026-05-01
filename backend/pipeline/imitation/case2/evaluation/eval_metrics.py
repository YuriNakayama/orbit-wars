"""Multi-faceted validation metrics for imitation/case2 IL heads.

Pure-computation helpers (ECE, F1, etc.) live in `src/evaluation/metrics.py`.
This wrapper wires the case2 model + dataset into those helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch.utils.data import DataLoader

from evaluation.metrics import (
    FromArrays,
    ShipsArrays,
    TargetArrays,
    from_metrics,
    ships_metrics,
    target_metrics,
)
from pipeline.imitation.case2.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case2.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case2.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case2.policy.types import BatchFeatures
from pipeline.imitation.case2.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)

logger = logging.getLogger(__name__)


def _to_batch_features(sample: BatchedSample) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats,
        planet_mask=sample.planet_mask,
        my_planet_mask=sample.my_planet_mask,
        target_mask=sample.target_mask,
        global_feats=sample.global_feats,
        template_ctx=sample.template_ctx,
    )


def collect_arrays(
    model: DeepSetsPolicy, loader: DataLoader[BatchedSample]
) -> tuple[FromArrays, TargetArrays, ShipsArrays]:
    """Run model in eval mode over loader, return per-head arrays."""
    model.eval()
    from_probs_list: list[np.ndarray] = []
    from_labels_list: list[np.ndarray] = []
    target_probs_list: list[np.ndarray] = []
    target_labels_list: list[np.ndarray] = []
    ships_probs_list: list[np.ndarray] = []
    ships_labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features = _to_batch_features(batch)
            output = model(features)
            from_logits = output.from_logits
            target_logits = output.target_logits
            ships_logits = output.ships_logits

            my_mask = batch.my_planet_mask
            from_multihot = batch.from_multihot

            valid = my_mask
            from_probs_full = torch.sigmoid(from_logits)
            from_probs_list.append(
                from_probs_full[valid].detach().cpu().numpy().astype(np.float32)
            )
            from_labels_list.append(
                from_multihot[valid].detach().cpu().numpy().astype(np.bool_)
            )

            fired = from_multihot & valid
            if fired.any():
                b_idx, p_idx = fired.nonzero(as_tuple=True)
                tp = (
                    torch.softmax(target_logits[b_idx, p_idx], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                tl = (
                    batch.target_per_src[b_idx, p_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                sp = (
                    torch.softmax(ships_logits[b_idx, p_idx], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                sl = (
                    batch.ships_per_src[b_idx, p_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                target_probs_list.append(tp)
                target_labels_list.append(tl)
                ships_probs_list.append(sp)
                ships_labels_list.append(sl)

    from_arr = FromArrays(
        probs=(
            np.concatenate(from_probs_list)
            if from_probs_list
            else np.zeros(0, dtype=np.float32)
        ),
        labels=(
            np.concatenate(from_labels_list)
            if from_labels_list
            else np.zeros(0, dtype=np.bool_)
        ),
    )
    if target_probs_list:
        target_arr = TargetArrays(
            probs=np.concatenate(target_probs_list, axis=0),
            labels=np.concatenate(target_labels_list, axis=0),
        )
        ships_arr = ShipsArrays(
            probs=np.concatenate(ships_probs_list, axis=0),
            labels=np.concatenate(ships_labels_list, axis=0),
        )
    else:
        target_arr = TargetArrays(
            probs=np.zeros((0, NUM_TEMPLATES), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )
        ships_arr = ShipsArrays(
            probs=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )
    return from_arr, target_arr, ships_arr


def compute_all_metrics(
    model: DeepSetsPolicy, loader: DataLoader[BatchedSample]
) -> dict[str, Any]:
    from_arr, target_arr, ships_arr = collect_arrays(model, loader)
    return {
        "from": from_metrics(from_arr),
        "target": target_metrics(target_arr, num_classes=NUM_TEMPLATES),
        "ships": ships_metrics(ships_arr),
    }


def _build_model(weights_path: Path, model_cfg: dict[str, Any]) -> DeepSetsPolicy:
    cfg = ModelConfig(
        planet_in_dim=int(model_cfg.get("planet_in_dim", PLANET_FEAT_DIM)),
        global_in_dim=int(model_cfg.get("global_in_dim", GLOBAL_FEAT_DIM)),
        hidden=int(model_cfg.get("hidden", 128)),
        ships_buckets=int(model_cfg.get("ships_buckets", 4)),
    )
    model = DeepSetsPolicy(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    weights: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case2/policy/weights.pt"),
        "--weights",
        "-w",
        help="Trained weights path",
    ),
    val: Path = typer.Option(  # noqa: B008
        Path("data/mart/imitation/case2/val.parquet"),
        "--val",
        "-v",
        help="Validation parquet path",
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case2/configs/il_baseline.yaml"),
        "--config",
        "-c",
        help="Config YAML (read for model dims)",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case2/evaluation/results_metrics.json"),
        "--out",
        "-o",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    label: str = typer.Option("", "--label", help="Free-form tag stored in JSON"),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with config.open() as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    model = _build_model(weights, model_cfg)
    ablation_cfg = (cfg.get("data", {}) or {}).get("ablation", {}) or {}
    ds = CaseThreeDataset(
        val,
        mask_planet_cols=list(ablation_cfg.get("planet_cols", []) or []),
        mask_global_cols=list(ablation_cfg.get("global_cols", []) or []),
    )
    loader: DataLoader[BatchedSample] = DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    metrics = compute_all_metrics(model, loader)
    metrics["weights"] = str(weights)
    metrics["val"] = str(val)
    if label:
        metrics["label"] = label
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    logger.info("metrics written to %s", out)


if __name__ == "__main__":
    app()
