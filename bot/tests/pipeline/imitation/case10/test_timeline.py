"""case5 timeline.py の sanity test (rulebase case6 portage)。"""

from __future__ import annotations

from pipeline.imitation.case10.policy.timeline import (
    DEFAULT_HORIZON,
    TimelinePlanet,
    simulate_planet_timeline,
    summarize_timeline,
)


def test_no_arrivals_planet_holds_stable() -> None:
    """到着 fleet なし → owner / ships は production で増えるだけ、陥落しない。"""
    planet = TimelinePlanet(id=1, owner=0, ships=10, production=2)
    timeline = simulate_planet_timeline(planet, [], player=0, horizon=10)

    assert timeline["owner_at"][0] == 0
    assert timeline["owner_at"][10] == 0
    assert timeline["fall_turn"] is None
    assert timeline["holds_full"] is True
    # ships=10 + 2*10 production = 30
    assert timeline["ships_at"][10] == 30.0


def test_overwhelming_enemy_predicts_fall() -> None:
    """陥落するシナリオで fall_turn と fall_predicted が立つ。"""
    planet = TimelinePlanet(id=1, owner=0, ships=5, production=1)
    # turn=2 に敵 100 ships 到着 → 陥落
    arrivals: list[tuple[float, int, int]] = [(2.0, 1, 100)]
    timeline = simulate_planet_timeline(planet, arrivals, player=0, horizon=10)

    assert timeline["fall_turn"] == 2
    assert timeline["owner_at"][2] == 1
    assert timeline["holds_full"] is False

    summary = summarize_timeline(timeline)
    assert summary["fall_predicted"] == 1.0
    # ttf_norm = 2/10 = 0.2
    assert summary["ttf_norm"] == 0.2


def test_summarize_no_fall_returns_clean_defaults() -> None:
    """陥落なしの場合、ttf_norm=1.0 / fall_predicted=0 / surplus>0。"""
    planet = TimelinePlanet(id=2, owner=0, ships=10, production=1)
    timeline = simulate_planet_timeline(planet, [], player=0, horizon=DEFAULT_HORIZON)
    summary = summarize_timeline(timeline)

    assert summary["fall_predicted"] == 0.0
    assert summary["ttf_norm"] == 1.0
    assert summary["surplus"] > 0
    assert summary["loss_3turn"] == 0.0  # production 単調増加なので 0


def test_keep_needed_binary_search_finds_minimum() -> None:
    """keep_needed は 2 分探索で defense に必要な最小値を返す。"""
    planet = TimelinePlanet(id=3, owner=0, ships=20, production=0)
    # turn=5 に敵 8 ships → keep_needed=8 (8 ships 残せば守れる)
    arrivals: list[tuple[float, int, int]] = [(5.0, 1, 8)]
    timeline = simulate_planet_timeline(planet, arrivals, player=0, horizon=10)

    assert timeline["holds_full"] is True
    assert timeline["keep_needed"] == 8


def test_neutral_planet_min_owned_zero() -> None:
    """planet が自所有でないと min_owned は 0、keep_needed も 0。"""
    planet = TimelinePlanet(id=4, owner=-1, ships=5, production=0)
    timeline = simulate_planet_timeline(planet, [], player=0, horizon=5)
    summary = summarize_timeline(timeline)

    assert summary["min_owned"] == 0.0
    assert summary["keep_needed"] == 0.0
    assert summary["fall_predicted"] == 0.0
