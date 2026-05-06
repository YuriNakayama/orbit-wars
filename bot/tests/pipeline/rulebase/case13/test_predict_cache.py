"""case13 predict cache 等価性テスト。

case13 の `predict_planet_position` は cache lookup を優先するが、
cache miss 時は元実装と同一値を返さなければならない。本 test は reference
実装 (case4 と同一の式) を inline で持ち、case13 出力と完全一致するかを検証する。
case4 module への直接依存はクロスケース独立ルール (`.claude/rules/bot/pipeline.md`)
に違反するので避ける。
"""

from __future__ import annotations

import math
import random

import pytest

from pipeline.rulebase.case13.baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    ROTATION_LIMIT,
)
from pipeline.rulebase.case13.baseline.core.physics import (
    predict_planet_position as cached_predict,
)
from pipeline.rulebase.case13.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case13.baseline.core.types import Planet


def _reference_predict(
    planet: Planet,
    initial_by_id: dict[int, Planet],
    angular_velocity: float,
    turns: int,
) -> tuple[float, float]:
    """case4 と同一式の参照実装。case13 cache 値との比較に使う。"""
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = math.hypot(init.x - CENTER_X, init.y - CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def _make_planet(pid: int, x: float, y: float, radius: float) -> Planet:
    return Planet(id=pid, owner=-1, x=x, y=y, radius=radius, ships=10, production=2)


@pytest.fixture(autouse=True)
def _clear_cache_each_test() -> None:
    reset_predict_cache()


def test_static_planet_returns_same_position() -> None:
    """ROTATION_LIMIT 越え (静的 planet) は turns に関係なく current pos を返す。

    ROTATION_LIMIT=50 なので CENTER (50,50) からの距離 r + radius >= 50
    を満たす配置にする。
    """
    p = _make_planet(0, x=98.0, y=50.0, radius=3.0)  # r=48, r+radius=51 >= 50
    initial = {0: p}
    cached = cached_predict(p, initial, 0.05, turns=10)
    ref = _reference_predict(p, initial, 0.05, turns=10)
    assert cached == ref
    assert cached == (98.0, 50.0)


def test_rotating_planet_matches_reference() -> None:
    """回転 planet の予測位置が reference 実装と完全一致する。"""
    rng = random.Random(42)
    cases: list[tuple[Planet, dict[int, Planet], float, int]] = []
    for i in range(50):
        ang = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(12.0, 25.0)
        init_x = CENTER_X + r * math.cos(ang)
        init_y = CENTER_Y + r * math.sin(ang)
        cur_ang = ang + rng.uniform(0, math.pi)
        cur_x = CENTER_X + r * math.cos(cur_ang)
        cur_y = CENTER_Y + r * math.sin(cur_ang)
        p_init = _make_planet(i, init_x, init_y, radius=2.5)
        p_cur = _make_planet(i, cur_x, cur_y, radius=2.5)
        cases.append((p_cur, {i: p_init}, rng.uniform(-0.1, 0.1), rng.randint(1, 110)))

    for p_cur, initial, av, turns in cases:
        cached_xy = cached_predict(p_cur, initial, av, turns)
        ref_xy = _reference_predict(p_cur, initial, av, turns)
        assert cached_xy == pytest.approx(ref_xy, abs=1e-12)


def test_cache_hit_returns_same_value_twice() -> None:
    p_init = _make_planet(7, 60.0, 50.0, 2.0)
    p_cur = _make_planet(7, 55.0, 55.0, 2.0)
    initial = {7: p_init}
    first = cached_predict(p_cur, initial, 0.03, 12)
    second = cached_predict(p_cur, initial, 0.03, 12)
    assert first == second


def test_reset_cache_drops_entries() -> None:
    """reset_predict_cache 後は同 turn でも再計算される (cache が空)。"""
    p_init = _make_planet(3, 60.0, 50.0, 2.0)
    p_cur = _make_planet(3, 55.0, 55.0, 2.0)
    initial = {3: p_init}
    cached_predict(p_cur, initial, 0.03, 5)
    from pipeline.rulebase.case13.baseline.core import physics as case13_physics

    assert len(case13_physics._PREDICT_PLANET_CACHE) > 0
    reset_predict_cache()
    assert len(case13_physics._PREDICT_PLANET_CACHE) == 0


def test_missing_initial_returns_current_pos() -> None:
    """initial_by_id に planet が無いなら current 位置を返す。"""
    p_cur = _make_planet(99, 55.0, 55.0, 2.0)
    cached_xy = cached_predict(p_cur, {}, 0.03, 5)
    ref_xy = _reference_predict(p_cur, {}, 0.03, 5)
    assert cached_xy == ref_xy == (55.0, 55.0)
