"""Integration test for imitation/case4 CandidatePolicy."""

from __future__ import annotations

import torch

from pipeline.imitation.case4.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case4.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
)
from pipeline.imitation.case4.policy.model import (
    CandidatePolicy,
    ModelConfig,
    count_parameters,
)


def _planet_row(pid: int, owner: int, x: float, y: float) -> list[float]:
    return [pid, owner, x, y, 1.5, 10, 1]


def test_model_forward_shapes() -> None:
    cfg = ModelConfig()
    model = CandidatePolicy(cfg)
    model.eval()
    assert count_parameters(model) > 0

    planets = [
        _planet_row(0, 0, 20.0, 50.0),
        _planet_row(1, 1, 80.0, 50.0),
        _planet_row(2, -1, 50.0, 20.0),
    ]
    obs = {
        "step": 1,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
    }
    batch, _ = featurize(obs, HistoryState())
    with torch.no_grad():
        output = model(batch)
    assert output.candidate_logits.shape == (1, MAX_PLANETS, CAND_K)


def test_invalid_slots_are_masked_with_large_negative() -> None:
    """For non-mine slots, candidate_mask is all False, so logits should be
    a large-magnitude negative (-1e9) that softmax treats as ~0 without
    overflowing log_softmax (finfo.min would overflow)."""
    cfg = ModelConfig()
    model = CandidatePolicy(cfg)
    model.eval()
    planets = [
        _planet_row(0, 0, 20.0, 50.0),
        _planet_row(1, 1, 80.0, 50.0),
    ]
    obs = {
        "step": 1,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
    }
    batch, snap = featurize(obs, HistoryState())
    with torch.no_grad():
        output = model(batch)
    # non-my planet — candidate_mask all-False → logits all -1e9
    for slot, pid in enumerate(snap.planet_ids):
        if pid in snap.my_planet_ids:
            continue
        for k in range(CAND_K):
            v = float(output.candidate_logits[0, slot, k].item())
            assert v < -1e8


def test_dimensions_match_config() -> None:
    cfg = ModelConfig(
        planet_in_dim=PLANET_FEAT_DIM,
        global_in_dim=GLOBAL_FEAT_DIM,
        cand_in_dim=CAND_FEAT_DIM,
        cand_k=CAND_K,
        hidden=64,
    )
    model = CandidatePolicy(cfg)
    assert model.cfg.hidden == 64
