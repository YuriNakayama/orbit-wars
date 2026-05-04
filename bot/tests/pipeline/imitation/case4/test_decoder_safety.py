"""Bug-reproduction tests for pipeline.imitation.case4.policy.decoder safety.

case4 differs from case1/2/3/5 (slot-based candidate decoder), so we drive
the safety filter via candidate_pid + manually constructed PolicyOutput
rather than templates. Same three bug classes.
"""

from __future__ import annotations

import math

import torch

from pipeline.imitation.case4.policy.candidates import CAND_K
from pipeline.imitation.case4.policy.decoder import decode
from pipeline.imitation.case4.policy.featurizer import HistoryState, featurize
from pipeline.imitation.case4.policy.types import PolicyOutput


def _planet_row(
    pid: int,
    owner: int,
    x: float,
    y: float,
    *,
    radius: float = 2.0,
    ships: int = 50,
    production: int = 1,
) -> list[float]:
    return [pid, owner, x, y, radius, ships, production]


def _force_slot1_fire(
    obs: dict[str, object],
) -> list[list[int | float]]:
    """Drive slot=1 (first candidate) for every src and run decode."""
    batch, snap = featurize(obs, HistoryState())
    logits = torch.full((1, batch.candidate_mask.shape[1], CAND_K), -10.0)
    logits[..., 1] = 5.0
    output = PolicyOutput(candidate_logits=logits)
    return decode(
        output,
        snap,
        obs,
        candidate_pid=batch.candidate_pid[0],
        candidate_mask=batch.candidate_mask[0],
    )


def test_bug2_high_angvel_orbital_target_action_is_dropped() -> None:
    """Bug 2: orbital target moves outside radius across ±tolerance."""
    obs: dict[str, object] = {
        "step": 5,
        "player": 0,
        "planets": [
            _planet_row(1, 0, 20.0, 10.0, radius=2.0, ships=50),
            _planet_row(2, 1, 70.0, 50.0, radius=1.0, ships=30),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            _planet_row(1, 0, 20.0, 10.0, radius=2.0, ships=50),
            _planet_row(2, 1, 70.0, 50.0, radius=1.0, ships=30),
        ],
        "angular_velocity": 0.5,
    }
    actions = _force_slot1_fire(obs)
    # Filter to actions originating from src 1 (the only my-planet).
    src1_actions = [a for a in actions if int(a[0]) == 1]
    assert src1_actions == [], (
        f"high-angvel miss action should be dropped, got {src1_actions}"
    )


def test_clean_short_range_shot_still_fires() -> None:
    """Regression guard: clean geometry must still emit an action."""
    obs: dict[str, object] = {
        "step": 5,
        "player": 0,
        "planets": [
            _planet_row(1, 0, 20.0, 15.0, radius=2.0, ships=50),
            _planet_row(2, 1, 40.0, 15.0, radius=2.0, ships=10),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            _planet_row(1, 0, 20.0, 15.0, radius=2.0, ships=50),
            _planet_row(2, 1, 40.0, 15.0, radius=2.0, ships=10),
        ],
        "angular_velocity": 0.0,
    }
    actions = _force_slot1_fire(obs)
    src1_actions = [a for a in actions if int(a[0]) == 1]
    assert len(src1_actions) >= 1
    src_id, angle, ships = src1_actions[0]
    assert src_id == 1
    assert math.isfinite(float(angle))
    assert int(ships) >= 1
