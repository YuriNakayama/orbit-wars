"""Unit tests for pipeline.imitation.case3.policy.decoder."""

from __future__ import annotations

import torch

from pipeline.imitation.case3.policy.decoder import decode
from pipeline.imitation.case3.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case3.policy.types import PolicyOutput, WorldSnapshot

NO_OP_LABEL = MAX_PLANETS


def _make_obs() -> dict[str, object]:
    return {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [],
        "planets": [
            [1, 0, 20.0, 20.0, 2.0, 50, 5],
            [2, 1, 80.0, 80.0, 2.0, 40, 4],
            [3, -1, 50.0, 30.0, 1.5, 10, 2],
        ],
        "fleets": [],
    }


def _make_output(
    from_logit_per_slot: list[float],
    target_slot_per_src: list[int],
    ships_bucket_per_src: list[int],
) -> PolicyOutput:
    """Build a minimal PolicyOutput with strong (one-hot-ish) preferences."""
    p = MAX_PLANETS
    from_logits = torch.full((1, p), -10.0)
    for i, v in enumerate(from_logit_per_slot):
        from_logits[0, i] = v
    target_logits = torch.full((1, p, p + 1), -10.0)
    for src, tgt in enumerate(target_slot_per_src):
        target_logits[0, src, tgt] = 10.0
    ships_logits = torch.full((1, p, 5), -10.0)
    for src, b in enumerate(ships_bucket_per_src):
        ships_logits[0, src, b] = 10.0
    return PolicyOutput(
        from_logits=from_logits,
        target_logits=target_logits,
        ships_logits=ships_logits,
    )


def test_threshold_blocks_no_action() -> None:
    obs = _make_obs()
    snap = WorldSnapshot(
        planet_ids=(1, 2, 3),
        my_planet_ids=(1,),
        player=0,
        step=5,
    )
    # from_logit at slot 0 = -1 → sigmoid ≈ 0.27 < 0.5
    out = _make_output(
        from_logit_per_slot=[-1.0],
        target_slot_per_src=[2] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert actions == []


def test_noop_target_skipped() -> None:
    obs = _make_obs()
    snap = WorldSnapshot(
        planet_ids=(1, 2, 3),
        my_planet_ids=(1,),
        player=0,
        step=5,
    )
    # high from_prob but target picks the no-op slot
    out = _make_output(
        from_logit_per_slot=[5.0],
        target_slot_per_src=[NO_OP_LABEL] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert actions == []


def test_valid_fire_emits_action() -> None:
    obs = _make_obs()
    snap = WorldSnapshot(
        planet_ids=(1, 2, 3),
        my_planet_ids=(1,),
        player=0,
        step=5,
    )
    # fire from slot 0 (planet 1) at slot 2 (planet 3, neutral) with bucket 2 (50%)
    out = _make_output(
        from_logit_per_slot=[5.0],
        target_slot_per_src=[2] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert len(actions) == 1
    src_pid, angle, ships = actions[0]
    assert src_pid == 1
    assert isinstance(angle, float)
    assert ships == int(0.5 * 50)  # bucket 2 → 50% of 50 ships


def test_decode_is_pure() -> None:
    obs = _make_obs()
    snap = WorldSnapshot(
        planet_ids=(1, 2, 3),
        my_planet_ids=(1,),
        player=0,
        step=5,
    )
    out = _make_output(
        from_logit_per_slot=[5.0],
        target_slot_per_src=[2] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    a = decode(out, snap, obs)
    b = decode(out, snap, obs)
    assert a == b
