"""Unit tests for case10 opponent_model pure-function library."""

from __future__ import annotations

from pipeline.rulebase.case10.baseline import opponent_model as om
from pipeline.rulebase.case10.baseline.core.types import Fleet, Planet


def _p(
    pid: int,
    owner: int,
    x: float = 0.0,
    y: float = 0.0,
    ships: int = 0,
    production: int = 1,
) -> Planet:
    return Planet(pid, owner, x, y, 2.5, ships, production)


def _f(
    fid: int,
    owner: int,
    x: float,
    y: float,
    angle: float,
    from_pid: int,
    ships: int,
) -> Fleet:
    return Fleet(fid, owner, x, y, angle, from_pid, ships)


def test_make_snapshot_captures_all_planets_and_fleets() -> None:
    planets = [_p(1, 0, ships=10), _p(2, 1, ships=5)]
    fleets = [_f(100, 1, 0.0, 0.0, 0.0, 2, 4)]

    snap = om.make_snapshot(step=3, planets=planets, fleets=fleets)

    assert snap.step == 3
    assert snap.planet_ships == {1: 10, 2: 5}
    assert snap.planet_owner == {1: 0, 2: 1}
    assert snap.fleet_ids == {100}


def test_detect_launches_returns_empty_when_no_delta() -> None:
    planets = [_p(1, 1, ships=10, production=1)]
    prev = om.TurnSnapshot(
        step=0,
        planet_ships={1: 9},
        planet_owner={1: 1},
        planet_prod={1: 1},
        fleet_ids=set(),
    )

    events = om.detect_launches(prev, 1, planets, [], viewer_player=0)

    assert events == []


def test_detect_launches_infers_enemy_launch_with_target() -> None:
    # fleet moves toward a my_planet well within OM_PREDICT_HORIZON (20 turns).
    planets = [
        _p(1, 1, x=0.0, y=0.0, ships=12, production=1),
        _p(50, 0, x=20.0, y=0.0, ships=10, production=1),
    ]
    fleets = [_f(fid=200, owner=1, x=5.0, y=0.0, angle=0.0, from_pid=1, ships=8)]
    prev = om.TurnSnapshot(
        step=0,
        planet_ships={1: 20, 50: 10},
        planet_owner={1: 1, 50: 0},
        planet_prod={1: 1, 50: 1},
        fleet_ids=set(),
    )

    events = om.detect_launches(prev, 1, planets, fleets, viewer_player=0)

    assert len(events) == 1
    assert events[0].owner == 1
    assert events[0].from_pid == 1
    assert events[0].target_class == "my_planet"


def test_detect_launches_skips_owner_changed() -> None:
    planets = [_p(1, 0, ships=5, production=1)]
    prev = om.TurnSnapshot(
        step=0,
        planet_ships={1: 20},
        planet_owner={1: 1},  # was enemy
        planet_prod={1: 1},
        fleet_ids=set(),
    )

    events = om.detect_launches(prev, 1, planets, [], viewer_player=0)

    assert events == []  # owner changed → combat ambiguity


def test_update_preferences_accumulates_with_decay() -> None:
    counts: dict[int, dict[str, float]] = {1: {"my_planet": 10.0}}
    events = [
        om.LaunchEvent(
            step=1,
            from_pid=2,
            owner=1,
            ships=5,
            target_pid=50,
            target_class="my_planet",
        )
    ]

    om.update_preferences(counts, events, decay=0.9)

    assert counts[1]["my_planet"] == 9.0 + 5.0


def test_preference_distribution_normalizes() -> None:
    counts = {1: {"my_planet": 3.0, "neutral_high": 1.0}}

    dist = om.preference_distribution(counts, 1)

    assert abs(dist["my_planet"] - 0.75) < 1e-9
    assert abs(dist["neutral_high"] - 0.25) < 1e-9


def test_preference_distribution_empty_owner() -> None:
    assert om.preference_distribution({}, 1) == {}


def test_compute_launch_rate_averages_over_window() -> None:
    events = [
        om.LaunchEvent(
            step=10, from_pid=1, owner=1, ships=5, target_pid=None, target_class="enemy"
        ),
        om.LaunchEvent(
            step=12, from_pid=1, owner=1, ships=5, target_pid=None, target_class="enemy"
        ),
    ]

    rate = om.compute_launch_rate(events, current_step=13, window=5)

    assert rate[1] == 2.0  # 10 ships / 5-turn window


def test_predict_future_arrivals_targets_nearest_my_planet() -> None:
    enemy = _p(1, 1, x=0.0, y=0.0, ships=15)
    mine = _p(50, 0, x=30.0, y=0.0, ships=5)
    launch_rate = {1: 4.0}
    pref_counts = {1: {"my_planet": 3.0, "enemy": 1.0}}

    preds = om.predict_future_arrivals(
        enemy_planets=[enemy],
        my_planets=[mine],
        launch_rate=launch_rate,
        pref_counts=pref_counts,
        horizon=20,
    )

    assert 50 in preds
    eta, owner, ships = preds[50][0]
    assert owner == 1
    assert ships >= 1
    assert 1 <= eta <= 20


def test_predict_future_arrivals_skips_low_stock() -> None:
    enemy = _p(1, 1, x=0.0, y=0.0, ships=1)
    mine = _p(50, 0, x=30.0, y=0.0, ships=5)

    preds = om.predict_future_arrivals(
        enemy_planets=[enemy],
        my_planets=[mine],
        launch_rate={1: 4.0},
        pref_counts={1: {"my_planet": 1.0}},
    )

    assert preds == {}


def test_is_new_game_detects_reset() -> None:
    prev = om.TurnSnapshot(
        step=100, planet_ships={}, planet_owner={}, planet_prod={}, fleet_ids=set()
    )

    assert om.is_new_game(None, 0)
    assert om.is_new_game(prev, 0)
    assert om.is_new_game(prev, 50)  # went backwards
    assert not om.is_new_game(prev, 101)


def test_trim_history_caps_entries() -> None:
    state = om.OMState()
    for i in range(700):
        state.launches.append(
            om.LaunchEvent(
                step=i,
                from_pid=1,
                owner=1,
                ships=1,
                target_pid=None,
                target_class="enemy",
            )
        )

    om.trim_history(state)

    assert len(state.launches) <= 600
