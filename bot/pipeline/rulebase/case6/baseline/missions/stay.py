"""STAY judge: per-source ship hold for defense reservation and burst-launch.

STAY is *not* a mission emitted into ``collect_missions``: emitting a Mission
requires a ShotOption that picks an action, and STAY's whole purpose is to
*suppress* actions on a turn. Instead, ``build_stay_holds`` returns a
``{src_id: held_ships}`` map that ``plan_moves`` uses to subtract from the
per-source attack budget before mission processing begins.

Two motives, both grounded in the user's hypothesis (2026-05-02):

1. **Defense hold** — already-launched enemy fleets create a near-future
   garrison deficit on one of our planets. Holding ships at a nearby
   source planet keeps them available for next-turn reinforcement and for
   the planet-resolution combat that occurs when those enemy fleets land.

2. **Burst hold** — fleet speed scales with ship count
   (``speed = 1 + (max-1) * (log(ships)/log(1000))^1.5``). Waiting one turn
   so production accumulates can produce a launch that arrives *earlier*
   than launching now with fewer ships, despite the one-turn delay.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import config as cfg
from ..core.physics import fleet_speed
from ..core.types import Planet
from ..core.world_model import WorldModel
from ..strategy_helpers import planet_distance


@dataclass(frozen=True)
class StayDecision:
    """Diagnostic record for a single STAY hold (one per src_id)."""

    src_id: int
    kind: str
    held_ships: int
    score: float
    reason: str


def _defense_risk_for_planet(world: WorldModel, planet: Planet) -> float:
    """Residual short-horizon enemy pressure on `planet`, beyond what reserve covers.

    Uses the precomputed ``base_timeline`` to find the worst-case lost garrison
    inside ``STAY_DEFENSE_HORIZON``: ``(min_owned)`` is what timeline already
    accounts for, but is computed against the existing reserve. The returned
    value is the additional ship deficit STAY would need to cover.
    """
    timeline = world.base_timeline.get(planet.id)
    if timeline is None:
        return 0.0
    horizon = min(int(timeline["horizon"]), cfg.STAY_DEFENSE_HORIZON)
    if horizon <= 0:
        return 0.0

    worst_deficit = 0.0
    for turn in range(1, horizon + 1):
        owner = timeline["owner_at"].get(turn)
        ships = float(timeline["ships_at"].get(turn, 0.0))
        if owner != world.player:
            # Planet would be lost by this turn under base trajectory.
            worst_deficit = max(worst_deficit, ships + 1.0)
            continue
        # Owner stays ours but garrison may dip dangerously close to zero.
        if ships < 1.0:
            worst_deficit = max(worst_deficit, 1.0 - ships)
    return worst_deficit


def _planet_value_weight(planet: Planet, world: WorldModel) -> float:
    """Production-weighted importance of a friendly planet (cheap proxy)."""
    base = float(max(1, int(planet.production)))
    base += world.indirect_wealth_map.get(planet.id, 0.0) * 0.05
    return base


def _build_defense_holds(
    world: WorldModel,
    decisions: list[StayDecision],
) -> dict[int, int]:
    """Compute defense-driven holds keyed by source planet id.

    For each threatened friendly planet, find nearby own planets within
    ``STAY_DEFENSE_MAX_TRAVEL_TURNS`` and assign a hold proportional to the
    deficit, weighted by planet value.
    """
    holds: dict[int, int] = {}
    if not cfg.STAY_DEFENSE_ENABLED or not world.my_planets:
        return holds

    threats: list[tuple[Planet, float]] = []
    for planet in world.my_planets:
        deficit = _defense_risk_for_planet(world, planet)
        if deficit <= 0.0:
            continue
        weighted = deficit * _planet_value_weight(planet, world)
        if weighted < cfg.STAY_DEFENSE_THRESHOLD:
            continue
        threats.append((planet, deficit))

    if not threats:
        return holds

    for threatened, deficit in threats:
        candidate_sources: list[tuple[Planet, int]] = []
        for src in world.my_planets:
            if src.id == threatened.id:
                continue
            eta = world.cached_travel_time(
                src.id, threatened.id, max(1, int(src.ships))
            )
            if eta > cfg.STAY_DEFENSE_MAX_TRAVEL_TURNS:
                continue
            usable = max(0, int(src.ships) - world.reserve.get(src.id, 0))
            if usable <= 0:
                continue
            candidate_sources.append((src, usable))

        if not candidate_sources:
            continue

        # Distribute the deficit greedily by proximity (cheaper ETA first).
        candidate_sources.sort(key=lambda item: planet_distance(item[0], threatened))
        remaining = int(deficit + 0.999)  # ceil
        for src, usable in candidate_sources:
            if remaining <= 0:
                break
            take = min(usable, remaining)
            if take <= 0:
                continue
            existing = holds.get(src.id, 0)
            holds[src.id] = existing + take
            remaining -= take
            decisions.append(
                StayDecision(
                    src_id=src.id,
                    kind="defense",
                    held_ships=take,
                    score=float(deficit),
                    reason=(f"defend planet={threatened.id} deficit={deficit:.1f}"),
                )
            )

    # Cap holds by actual usable inventory (reserve is independent).
    for src_id in list(holds.keys()):
        usable_now = max(
            0, int(world.planet_by_id[src_id].ships) - world.reserve.get(src_id, 0)
        )
        holds[src_id] = min(holds[src_id], usable_now)
        if holds[src_id] == 0:
            del holds[src_id]
    return holds


def _best_burst_target(
    src: Planet,
    world: WorldModel,
    ships_now: int,
    ships_next: int,
) -> tuple[Planet, int] | None:
    """Pick the target with the largest ETA gain from holding 1 turn at src.

    Returns (target, gain_turns) or None when no target would meaningfully
    benefit from a burst hold.
    """
    if ships_now < cfg.STAY_BURST_MIN_SHIPS:
        return None
    speed_now = fleet_speed(ships_now)
    speed_next = fleet_speed(ships_next)
    if speed_next <= speed_now:
        return None

    best: tuple[Planet, int] | None = None
    for target in world.planets:
        if target.id == src.id:
            continue
        if target.owner == world.player:
            continue
        # Use cheap analytic travel_time twice so we never re-enter
        # the heavy aim_with_prediction loop just to score a hold.
        # Cache hits across (src, target) pairs reused by other STAY motives.
        eta_now = world.cached_travel_time(src.id, target.id, ships_now)
        eta_next = world.cached_travel_time(src.id, target.id, ships_next)
        if eta_now >= 10**8 or eta_next >= 10**8:
            continue
        if eta_now > cfg.STAY_BURST_MAX_TARGET_TURNS:
            continue
        # Net arrival turn comparison: launching now arrives at eta_now,
        # holding 1 turn then launching arrives at eta_next + 1.
        gain = eta_now - (eta_next + 1)
        if gain < cfg.STAY_BURST_MIN_GAIN:
            continue
        if best is None or gain > best[1]:
            best = (target, gain)
    return best


def _build_burst_holds(
    world: WorldModel,
    decisions: list[StayDecision],
    consecutive_holds: dict[int, int] | None,
) -> dict[int, int]:
    """Compute burst-driven holds keyed by source planet id.

    When ``consecutive_holds`` is provided, burst holds are suppressed for any
    source that has already been burst-held ``STAY_BURST_MAX_HOLD_TURNS`` turns
    in a row. The suppressed source is recorded as a diagnostic decision so the
    cap is observable.
    """
    holds: dict[int, int] = {}
    if not cfg.STAY_BURST_ENABLED or not world.my_planets:
        return holds

    for src in world.my_planets:
        if src.id in world.doomed_candidates:
            continue
        usable_now = max(0, int(src.ships) - world.reserve.get(src.id, 0))
        if usable_now < cfg.STAY_BURST_MIN_SHIPS:
            continue
        ships_next = usable_now + max(0, int(src.production))

        decision = _best_burst_target(src, world, usable_now, ships_next)
        if decision is None:
            continue
        target, gain = decision

        if consecutive_holds is not None:
            already_held = consecutive_holds.get(src.id, 0)
            if already_held >= cfg.STAY_BURST_MAX_HOLD_TURNS:
                decisions.append(
                    StayDecision(
                        src_id=src.id,
                        kind="burst_capped",
                        held_ships=0,
                        score=float(gain),
                        reason=(
                            f"burst capped at MAX_HOLD_TURNS="
                            f"{cfg.STAY_BURST_MAX_HOLD_TURNS} "
                            f"(held {already_held}t toward planet={target.id})"
                        ),
                    )
                )
                continue

        holds[src.id] = usable_now
        decisions.append(
            StayDecision(
                src_id=src.id,
                kind="burst",
                held_ships=usable_now,
                score=float(gain),
                reason=(
                    f"burst toward planet={target.id} "
                    f"speed_gain={gain}t ships {usable_now}->{ships_next}"
                ),
            )
        )
    return holds


def build_stay_holds(
    world: WorldModel,
    consecutive_holds: dict[int, int] | None = None,
) -> tuple[dict[int, int], list[StayDecision]]:
    """Return (per-source hold map, ordered diagnostic decisions).

    Defense and burst can both fire on the same source; the larger hold wins.

    ``consecutive_holds`` (optional) maps src_id -> number of consecutive turns
    that source has been burst-held up to (but not including) the current turn.
    When provided, burst holds are suppressed for sources at or above
    ``STAY_BURST_MAX_HOLD_TURNS``. Defense holds are *not* gated by this cap.
    """
    decisions: list[StayDecision] = []
    if not cfg.STAY_ENABLED:
        return {}, decisions

    defense_holds = _build_defense_holds(world, decisions)
    burst_holds = _build_burst_holds(world, decisions, consecutive_holds)

    merged: dict[int, int] = dict(defense_holds)
    for src_id, ships in burst_holds.items():
        merged[src_id] = max(merged.get(src_id, 0), ships)
    return merged, decisions


__all__ = ["StayDecision", "build_stay_holds"]
