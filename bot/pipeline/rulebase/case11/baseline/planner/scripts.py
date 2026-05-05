"""Script library for case11 Portfolio Greedy Search.

A "script" maps (source planet, world) -> a single move (or None for hold).
Each script encodes one "role" the source planet might take this turn.
The PGS hill climbing assigns one script per source and the resulting move
list is the agent's action for the turn.

Scripts are pure functions that read the WorldModel; they do NOT mutate state.
The PGS evaluator simulates outcomes via `simulate_planet_timeline`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ..core.physics import fleet_speed
from ..core.types import Planet
from ..core.world_model import WorldModel

# A script returns either a move [src_id, angle, ships] or None for "hold".
ScriptMove = list[float] | None
ScriptFn = Callable[[Planet, WorldModel], ScriptMove]


@dataclass(frozen=True)
class Script:
    name: str
    fn: ScriptFn


# ---- Helpers --------------------------------------------------------------


def _angle_to(src: Planet, target: Planet) -> float:
    return math.atan2(target.y - src.y, target.x - src.x)


def _distance(src: Planet, target: Planet) -> float:
    return math.hypot(target.x - src.x, target.y - src.y)


def _eta_turns(src: Planet, target: Planet, ships: int) -> int:
    speed = fleet_speed(max(1, ships))
    return max(1, int(math.ceil(_distance(src, target) / max(0.01, speed))))


def _ships_to_capture_at(target: Planet, eta: int, world: WorldModel) -> int:
    """Approximate ships needed to capture target at arrival time `eta`.

    Falls back to garrison + production*eta + safety_1 when world doesn't
    expose the heavier `ships_needed_to_capture` cleanly.
    """
    try:
        return int(world.ships_needed_to_capture(target.id, eta, {}))
    except Exception:
        garrison = max(0, int(target.ships))
        prod_growth = int(max(0, target.production)) * eta if target.owner != -1 else 0
        return garrison + prod_growth + 1


# ---- Scripts --------------------------------------------------------------


def script_idle(src: Planet, world: WorldModel) -> ScriptMove:
    """Hold all ships at the source. Useful for planets under threat or
    that have no productive target this turn."""
    return None


def script_capture_safe(src: Planet, world: WorldModel) -> ScriptMove:
    """Send `need + safety_buffer` ships to the nearest neutral target.

    iter1 v2: send_cap = need + 4 (buffer for enemy reaction) instead of
    just `need`, so capture wins reliably and PGS evaluator differentiates
    this from idle.
    """
    if src.ships <= 0:
        return None
    best: tuple[Planet, int, float] | None = None  # (target, send, score)
    for target in world.planets:
        if target.id == src.id or target.owner == world.player:
            continue
        if target.owner != -1:
            continue  # only neutral
        eta = _eta_turns(src, target, src.ships)
        need = _ships_to_capture_at(target, eta, world)
        send = min(int(src.ships), need + 4)  # iter1 v2: +4 safety buffer
        if send <= 0 or send > src.ships:
            continue
        # Score: prefer near targets with high production
        score = float(target.production) / max(1, eta)
        if best is None or score > best[2]:
            best = (target, send, score)
    if best is None:
        return None
    target, send, _ = best
    return [src.id, _angle_to(src, target), send]


def script_capture_aggressive(src: Planet, world: WorldModel) -> ScriptMove:
    """Send most available ships to the highest-value reachable target,
    keeping a minimum reserve at the source for defense.

    iter1 v3 fix: reserve = max(5, prod * 3) to prevent home-fall after
    full-commit capture. v0/v1/v2 sent all ships and lost home reliably.
    """
    if src.ships <= 0:
        return None
    reserve = max(5, int(src.production) * 3)
    available = max(0, int(src.ships) - reserve)
    if available <= 0:
        return None
    best: tuple[Planet, float] | None = None
    for target in world.planets:
        if target.id == src.id or target.owner == world.player:
            continue
        eta = _eta_turns(src, target, available)
        need = _ships_to_capture_at(target, eta, world)
        if need > available:
            continue
        # Score by target production, with a soft preference for enemies.
        bonus = 1.5 if target.owner not in (-1, world.player) else 1.0
        score = bonus * float(target.production) / max(1, eta)
        if best is None or score > best[1]:
            best = (target, score)
    if best is None:
        return None
    return [src.id, _angle_to(src, best[0]), available]


def script_snipe(src: Planet, world: WorldModel) -> ScriptMove:
    """Intercept an enemy fleet en-route to a friendly target.

    Picks the closest enemy fleet whose target (best-effort heuristic) is a
    self-owned planet, then aims at the fleet's predicted intercept point.
    For v0 simplicity we just shoot at the fleet's current position.
    """
    if src.ships <= 0 or not world.fleets:
        return None
    best_fleet = None
    best_score = 0.0
    for fleet in world.fleets:
        if fleet.owner == world.player:
            continue
        # ETA from src to the fleet (treat fleet as a point target).
        dist = math.hypot(fleet.x - src.x, fleet.y - src.y)
        if dist <= 0.5:
            continue
        score = float(fleet.ships) / max(0.5, dist)
        if score > best_score:
            best_score = score
            best_fleet = fleet
    if best_fleet is None:
        return None
    angle = math.atan2(best_fleet.y - src.y, best_fleet.x - src.x)
    send = min(int(src.ships), max(1, int(best_fleet.ships) + 2))
    return [src.id, angle, send]


def script_harass(src: Planet, world: WorldModel) -> ScriptMove:
    """Send a medium fleet to drain enemy production. Targets the highest-
    production enemy planet within reach.

    iter1 v2: send_cap = src.ships // 2 (was //4). The smaller fleet was
    reliably losing to enemy reactions and PGS evaluator was treating
    "send 2" as nearly free, biasing it as the chosen script.
    """
    if src.ships < 4:
        return None
    best: tuple[Planet, float] | None = None
    for target in world.planets:
        if target.owner in (-1, world.player):
            continue
        eta = _eta_turns(src, target, max(1, int(src.ships) // 2))
        score = float(target.production) / max(1, eta)
        if best is None or score > best[1]:
            best = (target, score)
    if best is None:
        return None
    target = best[0]
    send = max(1, min(int(src.ships) // 2, int(src.ships) - 2))
    if send < 1:
        return None
    return [src.id, _angle_to(src, target), send]


def script_reinforce(src: Planet, world: WorldModel) -> ScriptMove:
    """Send ships to the most threatened friendly planet (excluding self)."""
    if src.ships < 2:
        return None
    best: tuple[Planet, float] | None = None
    for target in world.planets:
        if target.id == src.id or target.owner != world.player:
            continue
        # Threat score: ships - garrison, higher = more threatened.
        threat = float(world.opponent_threat_score.get(target.id, 0.0))
        if threat <= 0:
            continue
        if best is None or threat > best[1]:
            best = (target, threat)
    if best is None:
        return None
    target = best[0]
    send = max(1, int(src.ships) // 2)
    return [src.id, _angle_to(src, target), send]


def script_consolidate(src: Planet, world: WorldModel) -> ScriptMove:
    """Send half the ships to a friendly planet with high production
    (case4 fleet_consolidation spirit). Picks a friendly target that is
    NOT the source, prefers higher-production friendlies."""
    if src.ships < 4:
        return None
    best: tuple[Planet, float] | None = None
    for target in world.planets:
        if target.id == src.id or target.owner != world.player:
            continue
        if target.production <= src.production:
            continue
        eta = _eta_turns(src, target, max(1, int(src.ships) // 2))
        score = float(target.production) / max(1, eta)
        if best is None or score > best[1]:
            best = (target, score)
    if best is None:
        return None
    target = best[0]
    send = max(1, int(src.ships) // 2)
    return [src.id, _angle_to(src, target), send]


# ---- Registry -------------------------------------------------------------


SCRIPTS: tuple[Script, ...] = (
    Script("idle", script_idle),
    Script("capture_safe", script_capture_safe),
    Script("capture_aggressive", script_capture_aggressive),
    Script("snipe", script_snipe),
    Script("harass", script_harass),
    Script("reinforce", script_reinforce),
    Script("consolidate", script_consolidate),
)

SCRIPT_BY_NAME: dict[str, Script] = {s.name: s for s in SCRIPTS}
