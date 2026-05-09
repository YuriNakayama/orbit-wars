"""Unit tests for the case1 planner subpackage (mission_resolver / finalizer)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pipeline.rulebase.case1.baseline.core.types import Mission, Planet, ShotOption
from pipeline.rulebase.case1.baseline.planner.finalizer import enforce_inventory_cap
from pipeline.rulebase.case1.baseline.planner.mission_resolver import (
    SINGLE_SOURCE_MISSION_KINDS,
    process_multi_source_mission,
    process_single_source_mission,
)


@dataclass
class _StubWorld:
    """Minimal WorldModel-shaped stub satisfying mission_resolver's needs."""

    player: int = 0
    planet_by_id: dict[int, Planet] = field(default_factory=dict)
    needed_to_capture: int = 5
    reinforcement_needed: int = 0

    def ships_needed_to_capture(
        self,
        target_id: int,
        arrival_turn: int,
        planned_commitments: dict[int, list[tuple[int, int, int]]],
    ) -> int:
        return self.needed_to_capture

    def reinforcement_needed_for(
        self,
        target_id: int,
        arrival_turn: int,
        planned_commitments: dict[int, list[tuple[int, int, int]]],
    ) -> int:
        return self.reinforcement_needed


def _planet(pid: int, *, owner: int = -1, ships: int = 0) -> Planet:
    return Planet(
        id=pid, owner=owner, x=0.0, y=0.0, radius=1.0, ships=ships, production=1
    )


def _option(
    src_id: int, target_id: int, *, send_cap: int = 10, turns: int = 5
) -> ShotOption:
    return ShotOption(
        score=1.0,
        src_id=src_id,
        target_id=target_id,
        angle=0.0,
        turns=turns,
        needed=send_cap,
        send_cap=send_cap,
        mission="capture",
    )


def _modes() -> dict[str, Any]:
    return {
        "is_finishing": False,
        "attack_margin_mult": 1.0,
        "is_dense": False,
    }


# ---------- SINGLE_SOURCE_MISSION_KINDS constant ----------


def test_single_source_kinds_includes_known_kinds() -> None:
    assert "single" in SINGLE_SOURCE_MISSION_KINDS
    assert "snipe" in SINGLE_SOURCE_MISSION_KINDS
    assert "reinforce" in SINGLE_SOURCE_MISSION_KINDS
    assert "crash_exploit" in SINGLE_SOURCE_MISSION_KINDS


def test_single_source_kinds_excludes_swarm() -> None:
    assert "swarm" not in SINGLE_SOURCE_MISSION_KINDS


# ---------- process_single_source_mission ----------


def test_single_source_skipped_when_no_inventory(
    append_recorder: tuple[
        list[tuple[int, float, int]], Callable[[int, float, int], int]
    ],
) -> None:
    moves, append_move = append_recorder
    target = _planet(2, owner=-1, ships=5)
    world = _StubWorld(planet_by_id={1: _planet(1, owner=0, ships=10), 2: target})
    mission = Mission(
        kind="single", score=1.0, target_id=2, turns=5, options=[_option(1, 2)]
    )

    process_single_source_mission(
        mission,
        target,
        world,  # type: ignore[arg-type]
        _modes(),
        defaultdict(list),
        source_inventory_left=lambda src: 0,
        source_attack_left=lambda src: 0,
        append_move=append_move,
    )
    assert moves == []


def test_single_source_emits_capture_move(
    append_recorder: tuple[
        list[tuple[int, float, int]], Callable[[int, float, int], int]
    ],
) -> None:
    """snipe kind avoids preferred_send → minimal world stub is enough."""
    moves, append_move = append_recorder
    target = _planet(2, owner=-1, ships=4)
    world = _StubWorld(
        planet_by_id={1: _planet(1, owner=0, ships=10), 2: target},
        needed_to_capture=5,
    )
    mission = Mission(
        kind="snipe", score=1.0, target_id=2, turns=5, options=[_option(1, 2)]
    )

    process_single_source_mission(
        mission,
        target,
        world,  # type: ignore[arg-type]
        _modes(),
        defaultdict(list),
        source_inventory_left=lambda src: 10,
        source_attack_left=lambda src: 10,
        append_move=append_move,
    )
    # snipe: send = missing (5) exactly.
    assert len(moves) == 1
    src_id, _, ships = moves[0]
    assert src_id == 1
    assert ships == 5


def test_single_source_skipped_when_already_captured(
    append_recorder: tuple[
        list[tuple[int, float, int]], Callable[[int, float, int], int]
    ],
) -> None:
    moves, append_move = append_recorder
    target = _planet(2, owner=-1, ships=4)
    world = _StubWorld(
        planet_by_id={1: _planet(1, owner=0, ships=10), 2: target},
        needed_to_capture=0,  # already accounted-for by other missions
    )
    mission = Mission(
        kind="single", score=1.0, target_id=2, turns=5, options=[_option(1, 2)]
    )

    process_single_source_mission(
        mission,
        target,
        world,  # type: ignore[arg-type]
        _modes(),
        defaultdict(list),
        source_inventory_left=lambda src: 10,
        source_attack_left=lambda src: 10,
        append_move=append_move,
    )
    assert moves == []


# ---------- process_multi_source_mission ----------


def test_multi_source_skipped_when_caps_under_need(
    append_recorder: tuple[
        list[tuple[int, float, int]], Callable[[int, float, int], int]
    ],
) -> None:
    moves, append_move = append_recorder
    target = _planet(3, owner=-1, ships=20)
    world = _StubWorld(
        planet_by_id={
            1: _planet(1, owner=0, ships=5),
            2: _planet(2, owner=0, ships=5),
            3: target,
        },
        needed_to_capture=20,
    )
    mission = Mission(
        kind="swarm",
        score=1.0,
        target_id=3,
        turns=5,
        options=[_option(1, 3, send_cap=5), _option(2, 3, send_cap=5)],
    )
    process_multi_source_mission(
        mission,
        target,
        world,  # type: ignore[arg-type]
        defaultdict(list),
        source_attack_left=lambda src: 5,
        append_move=append_move,
    )
    assert moves == []


def test_multi_source_emits_two_moves_when_caps_cover_need(
    append_recorder: tuple[
        list[tuple[int, float, int]], Callable[[int, float, int], int]
    ],
) -> None:
    moves, append_move = append_recorder
    target = _planet(3, owner=-1, ships=8)
    world = _StubWorld(
        planet_by_id={
            1: _planet(1, owner=0, ships=10),
            2: _planet(2, owner=0, ships=10),
            3: target,
        },
        needed_to_capture=12,
    )
    mission = Mission(
        kind="swarm",
        score=1.0,
        target_id=3,
        turns=5,
        options=[
            _option(1, 3, send_cap=10, turns=4),
            _option(2, 3, send_cap=10, turns=5),
        ],
    )
    process_multi_source_mission(
        mission,
        target,
        world,  # type: ignore[arg-type]
        defaultdict(list),
        source_attack_left=lambda src: 10,
        append_move=append_move,
    )
    assert len(moves) >= 1
    sent_total = sum(m[2] for m in moves)
    assert sent_total >= 12


# ---------- enforce_inventory_cap ----------


def test_enforce_inventory_cap_drops_overspend() -> None:
    src = _planet(1, owner=0, ships=5)
    world = _StubWorld(planet_by_id={1: src})
    raw_moves: list[list[int | float]] = [
        [1, 0.0, 4],
        [1, 0.5, 4],  # Cumulative 8 > inventory 5: this should shrink.
    ]
    final = enforce_inventory_cap(raw_moves, world)  # type: ignore[arg-type]
    assert sum(int(m[2]) for m in final) == 5


def test_enforce_inventory_cap_drops_zero_send() -> None:
    src = _planet(1, owner=0, ships=5)
    world = _StubWorld(planet_by_id={1: src})
    raw_moves: list[list[int | float]] = [
        [1, 0.0, 5],
        [1, 0.5, 0],  # zero: should be dropped after cap, not in output.
    ]
    final = enforce_inventory_cap(raw_moves, world)  # type: ignore[arg-type]
    assert len(final) == 1
    assert int(final[0][2]) == 5


def test_enforce_inventory_cap_passes_through_under_cap() -> None:
    src1 = _planet(1, owner=0, ships=10)
    src2 = _planet(2, owner=0, ships=10)
    world = _StubWorld(planet_by_id={1: src1, 2: src2})
    raw_moves: list[list[int | float]] = [
        [1, 0.0, 4],
        [2, 0.5, 6],
    ]
    final = enforce_inventory_cap(raw_moves, world)  # type: ignore[arg-type]
    assert final == [[1, 0.0, 4], [2, 0.5, 6]]
