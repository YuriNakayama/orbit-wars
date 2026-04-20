"""Unit tests for pipeline.imitation.case1.policy.decoder."""

from __future__ import annotations

import torch

from pipeline.imitation.case1.policy.decoder import decode
from pipeline.imitation.case1.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case1.policy.templates import (
    NUM_TEMPLATES,
    T_NEAREST_NEUTRAL_LOW,
    T_NO_OP,
)
from pipeline.imitation.case1.policy.types import PolicyOutput, WorldSnapshot


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
    template_per_src: list[int],
    ships_bucket_per_src: list[int],
) -> PolicyOutput:
    """Build a minimal PolicyOutput with strong (one-hot-ish) preferences."""
    p = MAX_PLANETS
    from_logits = torch.full((1, p), -10.0)
    for i, v in enumerate(from_logit_per_slot):
        from_logits[0, i] = v
    target_logits = torch.full((1, p, NUM_TEMPLATES), -10.0)
    for src, tid in enumerate(template_per_src):
        target_logits[0, src, tid] = 10.0
    ships_logits = torch.full((1, p, 4), -10.0)
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
    out = _make_output(
        from_logit_per_slot=[-1.0],
        template_per_src=[T_NEAREST_NEUTRAL_LOW] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5, min_fire_topk=0)
    assert actions == []


def test_noop_template_skipped() -> None:
    obs = _make_obs()
    snap = WorldSnapshot(
        planet_ids=(1, 2, 3),
        my_planet_ids=(1,),
        player=0,
        step=5,
    )
    out = _make_output(
        from_logit_per_slot=[5.0],
        template_per_src=[T_NO_OP] + [0] * (MAX_PLANETS - 1),
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
    # NEAREST_NEUTRAL_LOW from planet 1 → planet 3 (only neutral, ships=10 ≤ src.50).
    # ships bucket 1 = 50% of 50 = 25 ships, capped to "need*2 = 22" by overfire
    # filter (target.ships+1 = 11 → need = 11, ships > need*2 → ships = 11).
    out = _make_output(
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_NEUTRAL_LOW] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[1] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert len(actions) == 1
    src_pid, angle, ships = actions[0]
    assert src_pid == 1
    assert isinstance(angle, float)
    assert ships == 11  # capped by overfire suppression to need (= 10+1)


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
        template_per_src=[T_NEAREST_NEUTRAL_LOW] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    a = decode(out, snap, obs)
    b = decode(out, snap, obs)
    assert a == b
