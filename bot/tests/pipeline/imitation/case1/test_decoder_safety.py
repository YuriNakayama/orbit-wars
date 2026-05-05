"""Bug-reproduction tests for pipeline.imitation.case1.policy.decoder safety.

The model's logits force the decoder into scenarios where the only viable
action is unsafe. With safety filtering wired in, the decoder should drop
(=> empty action list).

These tests cover the three bug classes reported from match logs:
  Bug 1: fleet trajectory grazes/enters the sun (segment check).
  Bug 2: orbital target leaves its radius across the ±tolerance window.
  Bug 3: comet expiry / non-target comet crossing.

State at TDD start (before trajectory_safety wiring):
  - Bug 1 cases already drop because aim_with_prediction returns None for
    sun-blocked geometry. We keep them as **regression guards** — the new
    safety filter must not regress these.
  - Bug 2 (high-angvel orbital target) is the **only genuine RED** case:
    the existing aim_with_prediction iterates and converges, then the decoder
    emits an action that misses because no ±tolerance radius check exists.
  - Bug 3a/3b currently drop because the target is unreachable in the existing
    geometry; they remain regression guards.
  - test_clean_short_range_shot_still_fires guards against over-filtering.
"""

from __future__ import annotations

import math

import torch

from pipeline.imitation.case1.policy.decoder import decode
from pipeline.imitation.case1.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case1.policy.templates import (
    NUM_TEMPLATES,
    T_NEAREST_ENEMY,
)
from pipeline.imitation.case1.policy.types import PolicyOutput, WorldSnapshot


def _make_output(
    *,
    n_sources: int,
    template_per_src: list[int],
    ships_bucket_per_src: list[int],
    from_logit_per_slot: list[float],
) -> PolicyOutput:
    """Build a strongly one-hot PolicyOutput for `n_sources` source slots."""
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
    _ = n_sources  # unused: bucket sizes pad to MAX_PLANETS
    return PolicyOutput(
        from_logits=from_logits,
        target_logits=target_logits,
        ships_logits=ships_logits,
    )


# ------------------------------------------------------------------ #
# Bug 1: sun consumption
# ------------------------------------------------------------------ #


def test_bug1_sun_blocked_only_target_emits_no_action() -> None:
    """Source on west edge, only enemy lies directly east through the sun.

    The straight-line shot must be filtered out (sun-blocked).
    """
    src_pid = 1
    enemy_pid = 2
    # Source at (20, 50), enemy at (80, 50) — the line through (50,50) hits the sun.
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [src_pid, 0, 20.0, 50.0, 2.0, 50, 5],
            [enemy_pid, 1, 80.0, 50.0, 2.0, 30, 4],
        ],
        "planets": [
            [src_pid, 0, 20.0, 50.0, 2.0, 50, 5],
            [enemy_pid, 1, 80.0, 50.0, 2.0, 30, 4],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, enemy_pid), my_planet_ids=(src_pid,), player=0, step=5
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    # Pre-fix: aim_with_prediction returns None (sun blocks segment) so this
    # case may already be empty. Make the assertion explicit anyway — once we
    # add the trajectory-sample check, near-grazes are also dropped.
    assert actions == []


def test_bug1_grazing_sun_within_safety_is_dropped() -> None:
    """Source/target geometry such that the segment passes JUST above the sun
    (within SUN_SAFETY but technically not "hit").

    Pre-fix: the segment-only check uses safety=1.5 -> tight grazes leak through.
    Post-fix: trajectory sampling at integer turn steps catches the same grazing.
    """
    src_pid = 1
    enemy_pid = 2
    # Source at (CENTER_X-30, CENTER_Y - SUN_R - 1.0) heading east at the enemy
    # placed at (CENTER_X+30, same y). Path passes 1 unit above sun edge — within
    # the SUN_R + SUN_SAFETY = 11.5 buffer.
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [src_pid, 0, 20.0, 39.0, 1.0, 50, 5],
            [enemy_pid, 1, 80.0, 39.0, 1.0, 30, 4],
        ],
        "planets": [
            [src_pid, 0, 20.0, 39.0, 1.0, 50, 5],
            [enemy_pid, 1, 80.0, 39.0, 1.0, 30, 4],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, enemy_pid), my_planet_ids=(src_pid,), player=0, step=5
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert actions == [], f"grazing-sun action should be dropped, got {actions}"


# ------------------------------------------------------------------ #
# Bug 2: orbital target moves out of radius across ±tolerance
# ------------------------------------------------------------------ #


def test_bug2_high_angvel_orbital_target_action_is_dropped() -> None:
    """Target is a small fast orbital — across ±1 turn tolerance the planet's
    centre moves more than radius * miss_radius_factor away from the predicted
    intercept point. The fleet would arrive and miss.
    """
    src_pid = 1
    tgt_pid = 2
    # Source far from sun (south side), enemy is a tiny fast orbital near the
    # source but rotating quickly. ang_vel large enough that the target's
    # angular displacement across ±1 turn = 2 * ang_vel * r > radius * 0.7.
    # r=20, ang_vel=0.5 rad/turn -> arc per turn = 10 units; target.radius=1.
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.5,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [src_pid, 0, 20.0, 10.0, 2.0, 50, 5],  # static-ish (far from center)
            [tgt_pid, 1, 70.0, 50.0, 1.0, 30, 4],  # orbital at r=20 from center
        ],
        "planets": [
            [src_pid, 0, 20.0, 10.0, 2.0, 50, 5],
            [tgt_pid, 1, 70.0, 50.0, 1.0, 30, 4],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, tgt_pid), my_planet_ids=(src_pid,), player=0, step=5
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    # Pre-fix: aim_with_prediction iterates and converges to some predicted
    # intercept; with no ±tolerance check the action is emitted and then misses.
    # Post-fix: intercept_holds_within_tolerance rejects.
    assert actions == [], f"high-angvel miss action should be dropped, got {actions}"


# ------------------------------------------------------------------ #
# Bug 3a: comet target expires before fleet arrives
# ------------------------------------------------------------------ #


def test_bug3a_comet_target_expires_before_arrival_is_dropped() -> None:
    """Target is a comet whose remaining path is shorter than the fleet's
    travel time. The fleet would arrive after the comet has despawned.
    """
    src_pid = 1
    comet_pid = 99
    # Build a comet path of length 4 starting at (60, 50). Source at (10, 50)
    # heading east; at speed 1 the fleet needs ~50 turns -> way past expiry.
    comet_path = [(60.0, 50.0), (61.0, 50.0), (62.0, 50.0), (63.0, 50.0)]
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [comet_pid],
        "comets": [
            {
                "planet_ids": [comet_pid],
                "paths": [comet_path],
                "path_index": 0,
                "radius": 1.0,
            }
        ],
        "initial_planets": [
            [src_pid, 0, 10.0, 5.0, 2.0, 50, 5],  # far south, static
            [comet_pid, 1, 60.0, 50.0, 1.0, 5, 1],
        ],
        "planets": [
            [src_pid, 0, 10.0, 5.0, 2.0, 50, 5],
            [comet_pid, 1, 60.0, 50.0, 1.0, 5, 1],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, comet_pid), my_planet_ids=(src_pid,), player=0, step=5
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert actions == [], f"comet-expiring action should be dropped, got {actions}"


# ------------------------------------------------------------------ #
# Bug 3b: fleet trajectory crosses a non-target comet's path
# ------------------------------------------------------------------ #


def test_bug3b_trajectory_crosses_non_target_comet_is_dropped() -> None:
    """Target is a static planet; a comet is traveling along a path that the
    fleet's trajectory crosses at the same time-step. The fleet would collide
    with the comet before reaching the target.
    """
    src_pid = 1
    enemy_pid = 2
    comet_pid = 99
    # Source at (10, 50). Static enemy at (90, 50). Comet path at (20..14, 50)
    # over 4 turns — the comet is moving westward into the fleet's trajectory.
    # Fleet at speed 1.0 (ships=10): turn 1 @(11,50), turn 2 @(12,50), ...
    # Comet at turn 3: (14, 50). Distance ~2 < comet.radius+1 -> collision.
    comet_path = [(20.0, 50.0), (17.0, 50.0), (14.0, 50.0), (13.0, 50.0)]
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [comet_pid],
        "comets": [
            {
                "planet_ids": [comet_pid],
                "paths": [comet_path],
                "path_index": 0,
                "radius": 2.0,
            }
        ],
        "initial_planets": [
            [src_pid, 0, 10.0, 50.0, 1.0, 50, 5],
            [enemy_pid, 1, 90.0, 50.0, 2.0, 30, 4],
        ],
        "planets": [
            [src_pid, 0, 10.0, 50.0, 1.0, 50, 5],
            [enemy_pid, 1, 90.0, 50.0, 2.0, 30, 4],
            [comet_pid, -1, 20.0, 50.0, 2.0, 5, 0],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, enemy_pid, comet_pid),
        my_planet_ids=(src_pid,),
        player=0,
        step=5,
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    # Note: with sun blocking the direct east path here too, this test relies
    # on the fact that the fleet trajectory crosses the comet BEFORE it would
    # reach the sun. The combined safety filter should drop the action.
    assert actions == [], (
        f"trajectory-comet-cross action should be dropped, got {actions}"
    )


# ------------------------------------------------------------------ #
# Sanity: a clean, safe shot still emits an action.
# ------------------------------------------------------------------ #


def test_clean_short_range_shot_still_fires() -> None:
    """Regression guard: when the geometry is safe (short range, no sun, no
    comets, static targets), the decoder must still emit an action.
    """
    src_pid = 1
    tgt_pid = 2
    # Both south of the sun on a clear east-west axis at y=15 (sun is at y=50).
    obs: dict[str, object] = {
        "player": 0,
        "step": 5,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [
            [src_pid, 0, 20.0, 15.0, 2.0, 50, 5],
            [tgt_pid, 1, 40.0, 15.0, 2.0, 10, 4],
        ],
        "planets": [
            [src_pid, 0, 20.0, 15.0, 2.0, 50, 5],
            [tgt_pid, 1, 40.0, 15.0, 2.0, 10, 4],
        ],
        "fleets": [],
    }
    snap = WorldSnapshot(
        planet_ids=(src_pid, tgt_pid), my_planet_ids=(src_pid,), player=0, step=5
    )
    out = _make_output(
        n_sources=1,
        from_logit_per_slot=[5.0],
        template_per_src=[T_NEAREST_ENEMY] + [0] * (MAX_PLANETS - 1),
        ships_bucket_per_src=[2] + [0] * (MAX_PLANETS - 1),
    )
    actions = decode(out, snap, obs, from_threshold=0.5)
    assert len(actions) == 1
    src_id, angle, ships = actions[0]
    assert src_id == src_pid
    assert math.isfinite(float(angle))
    assert int(ships) >= 1
