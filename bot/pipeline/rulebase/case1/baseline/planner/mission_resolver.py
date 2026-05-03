"""Resolve a single Mission into concrete moves (single/multi-source variants)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.config import REINFORCE_SAFETY_MARGIN
from ..core.types import Mission, Planet
from ..core.world_model import WorldModel
from ..strategy_helpers import preferred_send

SINGLE_SOURCE_MISSION_KINDS: frozenset[str] = frozenset(
    {"single", "snipe", "reinforce", "crash_exploit"}
)


def process_single_source_mission(
    mission: Mission,
    target: Planet,
    world: WorldModel,
    modes: dict[str, Any],
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    source_inventory_left: Callable[[int], int],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    """Append a single-source move (if all guards pass) and update commitments."""
    option = mission.options[0]

    left = (
        source_inventory_left(option.src_id)
        if mission.kind == "reinforce"
        else source_attack_left(option.src_id)
    )
    if left <= 0:
        return

    arrival_turn = option.turns

    if mission.kind == "reinforce":
        missing = world.reinforcement_needed_for(
            option.target_id, arrival_turn, planned_commitments
        )
    else:
        missing = world.ships_needed_to_capture(
            target.id, arrival_turn, planned_commitments
        )
    if missing <= 0:
        return

    send_limit = min(left, option.send_cap)
    if send_limit < missing:
        return

    if mission.kind in ("snipe", "crash_exploit"):
        send = missing
    elif mission.kind == "reinforce":
        send = min(send_limit, missing + REINFORCE_SAFETY_MARGIN)
    else:
        send = min(
            send_limit,
            max(
                missing,
                preferred_send(
                    target,
                    missing,
                    arrival_turn,
                    send_limit,
                    world,
                    modes,
                ),
            ),
        )
    if send < missing:
        return

    sent = append_move(option.src_id, option.angle, send)
    if sent < missing:
        return
    planned_commitments[target.id].append((arrival_turn, world.player, int(sent)))


def process_multi_source_mission(
    mission: Mission,
    target: Planet,
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    """Allocate multi-source ships and append concurrent moves."""
    limits: list[int] = []
    for option in mission.options:
        left = source_attack_left(option.src_id)
        limits.append(min(left, option.send_cap))
    if min(limits) <= 0:
        return

    missing = world.ships_needed_to_capture(
        target.id, mission.turns, planned_commitments
    )
    if missing <= 0:
        return
    if sum(limits) < missing:
        return

    ordered = sorted(
        zip(mission.options, limits, strict=True),
        key=lambda item: (item[0].turns, -item[1], item[0].src_id),
    )
    remaining = missing
    sends: dict[int, int] = {}
    for idx, (option, limit) in enumerate(ordered):
        remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
        send = min(limit, max(0, remaining - remaining_other))
        sends[option.src_id] = send
        remaining -= send
    if remaining > 0:
        return

    committed: list[tuple[int, int, int]] = []
    for option, _ in ordered:
        send = sends.get(option.src_id, 0)
        if send <= 0:
            continue
        actual = append_move(option.src_id, option.angle, send)
        if actual <= 0:
            continue
        committed.append((option.turns, world.player, int(actual)))
    if sum(item[2] for item in committed) < missing:
        return
    planned_commitments[target.id].extend(committed)
