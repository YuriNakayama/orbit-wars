"""Decoder dispatch test for imitation/case9 head variants."""

from __future__ import annotations

import pytest
import torch

from pipeline.imitation.case9.policy.decoder import decode
from pipeline.imitation.case9.policy.featurizer import HistoryState, featurize
from pipeline.imitation.case9.policy.model import (
    SUPPORTED_HEAD_MODES,
    Case9Policy,
    ModelConfig,
)


def _obs() -> dict[str, object]:
    return {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.05,
        "planets": [
            [0, 0, 30.0, 50.0, 3.0, 30, 1],  # mine
            [1, 1, 70.0, 50.0, 3.0, 10, 1],  # enemy
            [2, -1, 50.0, 30.0, 3.0, 5, 1],  # neutral
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [0, 0, 30.0, 50.0, 3.0, 30, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
        ],
    }


@pytest.mark.parametrize("head_mode", SUPPORTED_HEAD_MODES)
def test_decode_returns_list(head_mode: str) -> None:
    """Each variant must return a list[list[...]] without raising."""
    history = HistoryState()
    obs = _obs()
    batch, snap = featurize(obs, history)
    cfg = ModelConfig(head_mode=head_mode)
    model = Case9Policy(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    actions = decode(
        out,
        snap,
        obs,
        candidate_pid=batch.candidate_pid[0],
        candidate_mask=batch.candidate_mask[0],
        head_mode=head_mode,
        template_ctx=batch.template_ctx[0],
    )
    assert isinstance(actions, list)
    for act in actions:
        assert isinstance(act, list)
        assert len(act) == 3  # [src_pid, angle, ships]


def test_decode_unknown_head_raises() -> None:
    history = HistoryState()
    obs = _obs()
    batch, snap = featurize(obs, history)
    cfg = ModelConfig(head_mode="candidate")
    model = Case9Policy(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    with pytest.raises(ValueError, match="unknown head_mode"):
        decode(
            out,
            snap,
            obs,
            candidate_pid=batch.candidate_pid[0],
            candidate_mask=batch.candidate_mask[0],
            head_mode="bogus",
        )
