"""Smoke tests for imitation/case4 decoder."""

from __future__ import annotations

import torch

from pipeline.imitation.case4.policy.candidates import CAND_K
from pipeline.imitation.case4.policy.decoder import decode
from pipeline.imitation.case4.policy.featurizer import (
    HistoryState,
    featurize,
)
from pipeline.imitation.case4.policy.types import PolicyOutput


def _planet_row(
    pid: int, owner: int, x: float, y: float, ships: int = 10
) -> list[float]:
    return [pid, owner, x, y, 1.5, ships, 1]


def _make_obs() -> dict[str, object]:
    # Place planets so the line-of-sight does NOT cross the sun (50,50) — sun
    # masking would otherwise invalidate slot 1.
    planets = [
        _planet_row(0, 0, 20.0, 20.0, ships=30),  # mine, lots of ships
        _planet_row(
            1, 1, 80.0, 20.0, ships=5
        ),  # enemy, vulnerable, same y -> below sun
        _planet_row(2, -1, 30.0, 20.0, ships=3),  # neutral, close
    ]
    return {
        "step": 1,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
    }


def test_no_op_when_slot_zero_argmax() -> None:
    obs = _make_obs()
    batch, snap = featurize(obs, HistoryState())
    # Build candidate_logits where slot 0 (no-op) is the argmax for every my planet.
    logits = torch.full((1, batch.candidate_mask.shape[1], CAND_K), -1.0)
    logits[..., 0] = 5.0  # slot 0 = no-op wins
    output = PolicyOutput(candidate_logits=logits)
    actions = decode(
        output,
        snap,
        obs,
        candidate_pid=batch.candidate_pid[0],
        candidate_mask=batch.candidate_mask[0],
    )
    assert actions == []


def test_fire_when_slot_one_argmax() -> None:
    obs = _make_obs()
    batch, snap = featurize(obs, HistoryState())
    # Pick slot 1 for every my planet (note: invalid slots are masked to -inf in
    # the model; here we set the logits manually post-featurize so the decoder
    # mask check still applies — but candidate_mask should allow a fire from src 0).
    logits = torch.full((1, batch.candidate_mask.shape[1], CAND_K), -10.0)
    logits[..., 1] = 5.0
    output = PolicyOutput(candidate_logits=logits)
    actions = decode(
        output,
        snap,
        obs,
        candidate_pid=batch.candidate_pid[0],
        candidate_mask=batch.candidate_mask[0],
    )
    # Source 0 has 30 ships, enemy at 5 → should fire (mask True for slot 1).
    assert any(int(a[0]) == 0 for a in actions)
