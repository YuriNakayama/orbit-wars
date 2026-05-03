"""Smoke test: Graph U-Net accepts phase1 dim inputs without code change."""

from __future__ import annotations

import torch

from pipeline.imitation.case2.policy.featurizer_phase1 import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize,
)
from pipeline.imitation.case2.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case2.policy.templates import NUM_TEMPLATES


def _make_obs() -> dict[str, object]:
    return {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.01,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [1, 0, 20.0, 20.0, 2.0, 50, 5],
            [2, 1, 80.0, 80.0, 2.0, 40, 4],
            [3, -1, 50.0, 50.0, 1.5, 10, 2],
        ],
        "planets": [
            [1, 0, 20.0, 20.0, 2.0, 50, 5],
            [2, 1, 80.0, 80.0, 2.0, 40, 4],
            [3, -1, 50.0, 50.0, 1.5, 10, 2],
        ],
        "fleets": [],
    }


def test_phase1_end_to_end_forward() -> None:
    cfg = ModelConfig(planet_in_dim=PLANET_FEAT_DIM, global_in_dim=GLOBAL_FEAT_DIM)
    model = DeepSetsPolicy(cfg)
    model.eval()
    batch, _ = featurize(_make_obs())
    with torch.no_grad():
        out = model(batch)
    assert out.from_logits.shape == (1, MAX_PLANETS)
    assert out.target_logits.shape == (1, MAX_PLANETS, NUM_TEMPLATES)
    assert out.ships_logits.shape == (1, MAX_PLANETS, cfg.ships_buckets)
    assert torch.isfinite(out.target_logits).all().item()


def test_phase1_state_dict_under_1mb() -> None:
    import io

    cfg = ModelConfig(planet_in_dim=PLANET_FEAT_DIM, global_in_dim=GLOBAL_FEAT_DIM)
    model = DeepSetsPolicy(cfg)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    assert buf.tell() < 1_000_000
