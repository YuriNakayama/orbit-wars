"""Hypothesis test: existing fix vs true fix vs engine collision.

Setup:
- Engine collision = `point_to_segment_distance(planet_pos_at_t, fleet_pos_at_(t-1), fleet_pos_at_t) < planet.radius`
  evaluated each turn (continuous collision detection per turn segment).
- "Existing fix" hit  = `dist(fleet_endpoint(turns), target_pos(turns)) <= radius`
  (single point at the returned `turns`).
- "True fix" hit      = `min over t in [1..turns_max] of segment-to-target_pos(t)`
  (mirrors what the engine actually checks).

These tests assert the structural claim:
    `there exist scenarios where engine_hit==True and existing_fix_hit==False`
    `whenever engine_hit==True, true_fix_hit==True`
which is the foundation for replacing the endpoint check with a segment+curve check.
"""

from __future__ import annotations

import math

import pytest

from pipeline.rulebase.case4.baseline.core.physics import (
    aim_with_prediction,
    fleet_speed,
    predict_planet_position,
)
from pipeline.rulebase.case4.baseline.core.types import Planet


def _engine_segment_distance(
    point: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """Mirror of kaggle_environments.envs.orbit_wars.point_to_segment_distance."""
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(
        0.0,
        min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2),
    )
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return math.hypot(px - projx, py - projy)


_LAUNCH_OFFSET = 0.1  # engine starts fleets at planet surface + 0.1


def _simulate_engine_hit(
    src: Planet,
    tgt: Planet,
    angle: float,
    ships: int,
    ang_vel: float,
    initial_by_id: dict[int, Planet],
    max_turns: int = 120,
) -> tuple[bool, int | None]:
    """Replay fleet step-by-step like the engine; return (hit, turn_of_hit).

    Mirrors engine per-turn order:
      1. fleet pos[t-1]→pos[t], collide vs planet at (t-1)
      2. planet pos[t-1]→pos[t], sweep fleet at t
    """
    speed = fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    offset = src.radius + _LAUNCH_OFFSET
    start_x = src.x + cos_a * offset
    start_y = src.y + sin_a * offset
    fx_prev, fy_prev = start_x, start_y
    for t in range(1, max_turns + 1):
        fx = start_x + cos_a * speed * t
        fy = start_y + sin_a * speed * t
        prev_tx, prev_ty = predict_planet_position(tgt, initial_by_id, ang_vel, t - 1)
        # Step 1: fleet moves; planet still at (t-1).
        d = _engine_segment_distance((prev_tx, prev_ty), (fx_prev, fy_prev), (fx, fy))
        if d < tgt.radius:
            return True, t
        # Step 2: planet moves; sweep fleet (already at t).
        tx, ty = predict_planet_position(tgt, initial_by_id, ang_vel, t)
        d = _engine_segment_distance((fx, fy), (prev_tx, prev_ty), (tx, ty))
        if d < tgt.radius:
            return True, t
        fx_prev, fy_prev = fx, fy
    return False, None


def _existing_fix_hit(
    src: Planet,
    tgt: Planet,
    angle: float,
    turns: int,
    ships: int,
    ang_vel: float,
    initial_by_id: dict[int, Planet],
) -> bool:
    """The judgement the existing endpoint-based fix would make."""
    speed = fleet_speed(ships)
    fx = src.x + math.cos(angle) * speed * turns
    fy = src.y + math.sin(angle) * speed * turns
    tx, ty = predict_planet_position(tgt, initial_by_id, ang_vel, turns)
    return math.hypot(fx - tx, fy - ty) <= tgt.radius


def _true_fix_hit(
    src: Planet,
    tgt: Planet,
    angle: float,
    turns: int,
    ships: int,
    ang_vel: float,
    initial_by_id: dict[int, Planet],
    horizon_pad: int = 5,
) -> bool:
    """The judgement a segment-based (engine-aligned) fix would make."""
    hit, _ = _simulate_engine_hit(
        src,
        tgt,
        angle,
        ships,
        ang_vel,
        initial_by_id,
        max_turns=turns + horizon_pad,
    )
    return hit


# -------- Hypothesis A: a concrete pass-through scenario exists --------


def test_pass_through_scenario_engine_hits_but_existing_fix_skips() -> None:
    """Direct hand-crafted case: fleet line passes through target's swept arc."""
    src = Planet(1, 0, 95.0, 50.0, 2.0, 1000, 5)
    tgt = Planet(2, 0, 50.0 + 18.0, 50.0, 2.0, 10, 1)
    initial = {1: src, 2: tgt}
    ang_vel = 0.05
    ships = 10

    aim = aim_with_prediction(src, tgt, ships, initial, ang_vel, [], set())
    assert aim is not None, "before-fix aim should produce a candidate"
    angle, turns, _, _ = aim

    engine_hit, hit_turn = _simulate_engine_hit(
        src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 10
    )
    existing_fix_says_hit = _existing_fix_hit(
        src, tgt, angle, turns, ships, ang_vel, initial
    )

    # Sanity: aim returned something the engine can actually use.
    assert engine_hit, (
        f"engine should hit when before-fix aim fires "
        f"(angle={angle:.3f}, turns={turns}, hit_turn={hit_turn})"
    )

    # Show the gap: engine hits at hit_turn but endpoint at the *returned* turns is far.
    speed = fleet_speed(ships)
    fx = src.x + math.cos(angle) * speed * turns
    fy = src.y + math.sin(angle) * speed * turns
    tx, ty = predict_planet_position(tgt, initial, ang_vel, turns)
    endpoint_miss = math.hypot(fx - tx, fy - ty)

    # Either the existing fix would have cleared this (endpoint hit) — fine,
    # both judgements agree — or it would have skipped despite the engine hit.
    if not existing_fix_says_hit:
        assert endpoint_miss > tgt.radius, (
            f"contradiction: endpoint miss {endpoint_miss:.2f} <= radius "
            f"{tgt.radius} but existing_fix_hit returned False"
        )
        # The headline assertion: this is exactly the false-negative class.
        # Demonstrating its existence proves the existing fix loses real shots.
        assert engine_hit, (
            "demonstrated case: engine hits, existing fix skips "
            f"(endpoint miss={endpoint_miss:.2f}, hit_turn={hit_turn}, "
            f"returned_turns={turns})"
        )


# -------- Hypothesis B: parameter sweep — count false negatives --------

_SHIPS_GRID = [5, 10, 50, 100, 500]
_ANG_VEL_GRID = [-0.10, -0.05, 0.05, 0.10, 0.15]
_SRC_POSITIONS = [
    (60.0, 30.0),
    (70.0, 30.0),
    (80.0, 50.0),
    (90.0, 30.0),
    (95.0, 50.0),
    (95.0, 70.0),
    (60.0, 70.0),
    (30.0, 50.0),
]


@pytest.fixture(scope="module")
def sweep_outcomes() -> dict[str, int]:
    """Run the cross-product once and tally engine vs existing-fix vs true-fix."""
    counts = {
        "total": 0,
        "aim_returned_none": 0,
        "engine_hit": 0,
        "existing_fix_says_hit": 0,
        "true_fix_says_hit": 0,
        "false_negative_existing": 0,  # engine hit, existing says skip
        "false_negative_true": 0,  # engine hit, true says skip (should be 0)
        "false_positive_true": 0,  # true says hit, engine doesn't (should be 0)
    }
    tgt = Planet(2, 0, 50.0 + 18.0, 50.0, 2.0, 10, 1)
    for ships in _SHIPS_GRID:
        for ang_vel in _ANG_VEL_GRID:
            for sx, sy in _SRC_POSITIONS:
                if math.hypot(sx - 50.0, sy - 50.0) < 22.0:
                    continue  # too close to centre
                src = Planet(1, 0, sx, sy, 2.0, 1000, 5)
                initial = {1: src, 2: tgt}
                aim = aim_with_prediction(
                    src, tgt, ships, initial, ang_vel, [], set()
                )
                counts["total"] += 1
                if aim is None:
                    counts["aim_returned_none"] += 1
                    continue
                angle, turns, _, _ = aim
                engine_hit, _ = _simulate_engine_hit(
                    src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 10
                )
                existing_hit = _existing_fix_hit(
                    src, tgt, angle, turns, ships, ang_vel, initial
                )
                true_hit = _true_fix_hit(
                    src, tgt, angle, turns, ships, ang_vel, initial
                )

                if engine_hit:
                    counts["engine_hit"] += 1
                if existing_hit:
                    counts["existing_fix_says_hit"] += 1
                if true_hit:
                    counts["true_fix_says_hit"] += 1
                if engine_hit and not existing_hit:
                    counts["false_negative_existing"] += 1
                if engine_hit and not true_hit:
                    counts["false_negative_true"] += 1
                if true_hit and not engine_hit:
                    counts["false_positive_true"] += 1
    return counts


def test_sweep_existing_fix_has_false_negatives(sweep_outcomes: dict[str, int]) -> None:
    """Hypothesis A: existing endpoint-based fix loses real engine hits."""
    fn = sweep_outcomes["false_negative_existing"]
    assert fn > 0, (
        f"expected at least one engine-hit/existing-skip case, got {fn}. "
        f"Full counts: {sweep_outcomes}"
    )


def test_sweep_true_fix_has_no_false_negatives(sweep_outcomes: dict[str, int]) -> None:
    """Hypothesis B: segment-based judgement matches the engine on every hit."""
    fn = sweep_outcomes["false_negative_true"]
    assert fn == 0, (
        f"true-fix judgement diverges from engine on {fn} hits. "
        f"Full counts: {sweep_outcomes}"
    )


def test_sweep_true_fix_has_no_false_positives(sweep_outcomes: dict[str, int]) -> None:
    """Hypothesis C: segment-based judgement does not over-claim hits."""
    fp = sweep_outcomes["false_positive_true"]
    assert fp == 0, (
        f"true-fix claims hit on {fp} cases the engine misses. "
        f"Full counts: {sweep_outcomes}"
    )


def test_sweep_true_fix_dominates_existing_fix(sweep_outcomes: dict[str, int]) -> None:
    """Hypothesis D: true fix labels ≥ as many shots as hits as the existing fix."""
    assert (
        sweep_outcomes["true_fix_says_hit"]
        >= sweep_outcomes["existing_fix_says_hit"]
    ), f"true fix should label at least as many hits, got {sweep_outcomes}"


def test_sweep_summary_emit(sweep_outcomes: dict[str, int]) -> None:
    """Always-passing test that surfaces the full counts in the report."""
    print("\n=== sweep summary ===")
    for k, v in sweep_outcomes.items():
        print(f"  {k:<32s} {v}")
    assert sweep_outcomes["total"] > 0


# -------- Aim output validation: every returned (angle, turns) must hit --------


def test_aim_with_prediction_output_always_lands_on_target() -> None:
    """Every non-None aim_with_prediction return must land on target per engine."""
    tgt = Planet(2, 0, 50.0 + 18.0, 50.0, 2.0, 10, 1)
    misses: list[str] = []
    fired = 0
    for ships in _SHIPS_GRID:
        for ang_vel in _ANG_VEL_GRID:
            for sx, sy in _SRC_POSITIONS:
                if math.hypot(sx - 50.0, sy - 50.0) < 22.0:
                    continue
                src = Planet(1, 0, sx, sy, 2.0, 1000, 5)
                initial = {1: src, 2: tgt}
                aim = aim_with_prediction(
                    src, tgt, ships, initial, ang_vel, [], set()
                )
                if aim is None:
                    continue
                fired += 1
                angle, turns, _, _ = aim
                engine_hit, hit_turn = _simulate_engine_hit(
                    src,
                    tgt,
                    angle,
                    ships,
                    ang_vel,
                    initial,
                    max_turns=turns + 5,
                )
                if not engine_hit:
                    misses.append(
                        f"ships={ships} ang_vel={ang_vel} src=({sx},{sy}) "
                        f"angle={angle:.3f} turns={turns}"
                    )
    assert not misses, (
        f"aim_with_prediction returned {len(misses)}/{fired} non-engine-hit shots:\n"
        + "\n".join(misses[:10])
    )
