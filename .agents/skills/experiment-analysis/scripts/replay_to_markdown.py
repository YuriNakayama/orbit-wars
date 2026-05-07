"""Convert Orbit Wars replay JSON(.gz) into a compact Markdown analysis file.

Used by the `experiment-analysis` skill (Phase 2). Stdout is intentionally minimal
so the calling LLM only ever reads the on-disk Markdown — never raw replay state.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Planet schema: [id, owner, x, y, radius, ships, production]
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = range(7)
# Fleet schema: [id, owner, x, y, angle, from_planet_id, ships]
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

NEUTRAL = -1
MAX_LINES_BUDGET = 600
MAX_BYTES_BUDGET = 40_000


@dataclass
class TurnState:
    turn: int
    my_planets: int
    op_planets: int
    my_ships: int
    op_ships: int
    my_prod: int
    op_prod: int
    my_in_flight: int
    op_in_flight: int
    planet_owner: dict[int, int]


@dataclass
class Event:
    turn: int
    side: str
    type: str
    detail: str


def load_replay(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


LOSS_LABEL_TOKENS = ("loss", "lose", "lost", "negative", "defeat")


def infer_viewpoint(replay: dict[str, Any], label: str | None = None) -> int:
    """Pick a sensible viewpoint player.

    The point of this skill is letting humans see the analysis from the
    *interesting* side: when the user labelled a replay as a loss they want
    to see why their bot lost, not how the opponent won. So if the label hints
    at a loss, snap the viewpoint to the loser; otherwise default to the
    winner so 'win' / 'highlight' replays show the agent that's working.
    """
    rewards = replay.get("rewards") or []
    if not rewards:
        return 0
    is_loss = label is not None and any(tok in label.lower() for tok in LOSS_LABEL_TOKENS)
    target_sign = -1 if is_loss else 1
    for i, r in enumerate(rewards):
        if r is not None and ((target_sign > 0 and r > 0) or (target_sign < 0 and r < 0)):
            return i
    for i, r in enumerate(rewards):
        if r is not None and r > 0:
            return i
    return 0


def extract_states(replay: dict[str, Any], pid: int) -> list[TurnState]:
    steps = replay["steps"]
    states: list[TurnState] = []
    for t, step in enumerate(steps):
        obs = step[0]["observation"]
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        if not planets:
            continue
        my_planets = sum(1 for p in planets if p[P_OWNER] == pid)
        op_planets = sum(1 for p in planets if p[P_OWNER] not in (pid, NEUTRAL))
        my_ships = sum(int(p[P_SHIPS]) for p in planets if p[P_OWNER] == pid)
        op_ships = sum(int(p[P_SHIPS]) for p in planets if p[P_OWNER] not in (pid, NEUTRAL))
        my_prod = sum(int(p[P_PROD]) for p in planets if p[P_OWNER] == pid)
        op_prod = sum(int(p[P_PROD]) for p in planets if p[P_OWNER] not in (pid, NEUTRAL))
        my_in_flight = sum(int(f[F_SHIPS]) for f in fleets if f[F_OWNER] == pid)
        op_in_flight = sum(int(f[F_SHIPS]) for f in fleets if f[F_OWNER] not in (pid, NEUTRAL))
        planet_owner = {int(p[P_ID]): int(p[P_OWNER]) for p in planets}
        states.append(
            TurnState(
                turn=int(obs.get("step", t)),
                my_planets=my_planets,
                op_planets=op_planets,
                my_ships=my_ships,
                op_ships=op_ships,
                my_prod=my_prod,
                op_prod=op_prod,
                my_in_flight=my_in_flight,
                op_in_flight=op_in_flight,
                planet_owner=planet_owner,
            )
        )
    return states


def detect_events(
    replay: dict[str, Any],
    states: list[TurnState],
    pid: int,
    ship_loss_abs: int,
    ship_loss_rel: float,
) -> list[Event]:
    events: list[Event] = []
    steps = replay["steps"]
    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        for planet_id, owner in curr.planet_owner.items():
            prev_owner = prev.planet_owner.get(planet_id)
            if prev_owner is None or prev_owner == owner:
                continue
            if owner == pid:
                events.append(
                    Event(curr.turn, "self", "planet_gain", f"planet#{planet_id} {prev_owner}→self")
                )
            elif prev_owner == pid:
                events.append(
                    Event(curr.turn, "self", "planet_loss", f"planet#{planet_id} self→{owner}")
                )
            elif owner != NEUTRAL:
                events.append(
                    Event(curr.turn, "opp", "planet_gain", f"planet#{planet_id} {prev_owner}→{owner}")
                )

        my_delta = curr.my_ships - prev.my_ships
        if -my_delta >= ship_loss_abs and prev.my_ships > 0 and (-my_delta) >= ship_loss_rel * prev.my_ships:
            events.append(
                Event(curr.turn, "self", "ship_loss_burst", f"ships {prev.my_ships}→{curr.my_ships} ({my_delta:+d})")
            )
        op_delta = curr.op_ships - prev.op_ships
        if -op_delta >= ship_loss_abs and prev.op_ships > 0 and (-op_delta) >= ship_loss_rel * prev.op_ships:
            events.append(
                Event(curr.turn, "opp", "ship_loss_burst", f"ships {prev.op_ships}→{curr.op_ships} ({op_delta:+d})")
            )

    for t, step in enumerate(steps):
        if t == 0 or t >= len(states):
            continue
        action = step[pid].get("action") or []
        if not action:
            continue
        prev_owner_map = states[t - 1].planet_owner if t - 1 < len(states) else {}
        prev_planets = (steps[t - 1][0]["observation"].get("planets") or []) if t - 1 >= 0 else []
        planet_xy = {int(p[P_ID]): (float(p[P_X]), float(p[P_Y])) for p in prev_planets}
        from_xy_by_id = planet_xy
        enemy_attacks = 0
        for act in action:
            if not isinstance(act, list) or len(act) < 3:
                continue
            from_id, angle, ships = int(act[0]), float(act[1]), int(act[2])
            origin = from_xy_by_id.get(from_id)
            if origin is None:
                continue
            ox, oy = origin
            best_id, best_dist = None, math.inf
            for tid, (tx, ty) in planet_xy.items():
                if tid == from_id:
                    continue
                dx, dy = tx - ox, ty - oy
                target_angle = math.atan2(dy, dx)
                diff = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi
                dist = math.hypot(dx, dy) * (1 + abs(diff))
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None and prev_owner_map.get(best_id, NEUTRAL) not in (pid, NEUTRAL):
                enemy_attacks += ships
        if enemy_attacks > 0:
            events.append(
                Event(states[t].turn, "self", "enemy_planet_attack", f"sent {enemy_attacks} ships at enemy planets")
            )

    events.sort(key=lambda e: (e.turn, e.side, e.type))
    return events


def turning_points(states: list[TurnState], top_k: int = 5) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        delta = (curr.my_ships - curr.op_ships) - (prev.my_ships - prev.op_ships)
        side = "self" if delta > 0 else "opp"
        note = f"my {prev.my_ships}→{curr.my_ships}, op {prev.op_ships}→{curr.op_ships}"
        rows.append((curr.turn, delta, side, note))  # type: ignore[arg-type]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return [(t, d, f"{s} | {n}") for (t, d, s, n) in rows[:top_k]]  # type: ignore[misc]


def render_markdown(
    *,
    label: str,
    replay_path: Path,
    replay: dict[str, Any],
    pid: int,
    states: list[TurnState],
    events: list[Event],
    include_full: bool,
) -> str:
    rewards = replay.get("rewards") or []
    statuses = replay.get("statuses") or []
    seed = replay_path.stem.split("_seed")[-1].split(".")[0] if "_seed" in replay_path.stem else "unknown"
    winner = next((i for i, r in enumerate(rewards) if r is not None and r > 0), None)
    final = states[-1] if states else None

    lines: list[str] = []
    lines.append(f"# Replay seed={seed} — {label}")
    lines.append("")
    lines.append("## Meta")
    lines.append(f"- file: `{replay_path.name}`")
    lines.append(f"- viewpoint: player {pid}")
    lines.append(f"- winner: player {winner}" if winner is not None else "- winner: (none)")
    lines.append(f"- rewards: {rewards}")
    lines.append(f"- statuses: {statuses}")
    lines.append(f"- total_turns: {final.turn if final else 0}")
    lines.append("")

    if final is not None:
        lines.append("## Headline stats (final)")
        lines.append("| metric | self | opponent |")
        lines.append("|---|---:|---:|")
        lines.append(f"| planets    | {final.my_planets} | {final.op_planets} |")
        lines.append(f"| ships      | {final.my_ships}   | {final.op_ships}   |")
        lines.append(f"| production | {final.my_prod}    | {final.op_prod}    |")
        lines.append(f"| in-flight  | {final.my_in_flight} | {final.op_in_flight} |")
        lines.append("")

    tps = turning_points(states, top_k=5)
    if tps:
        lines.append("## Turning points (top 5 by ship-margin delta)")
        lines.append("| turn | delta | side | note |")
        lines.append("|---:|---:|---|---|")
        for turn, delta, sidenote in tps:
            side, note = sidenote.split(" | ", 1)
            lines.append(f"| {turn} | {delta:+d} | {side} | {note} |")
        lines.append("")

    if events:
        lines.append(f"## Key events ({len(events)})")
        lines.append("| turn | side | type | detail |")
        lines.append("|---:|---|---|---|")
        for ev in events:
            lines.append(f"| {ev.turn} | {ev.side} | {ev.type} | {ev.detail} |")
        lines.append("")

    if include_full and states:
        lines.append("<details><summary>Full per-turn stats</summary>")
        lines.append("")
        lines.append("| turn | mp | op | ms | os | mpr | opr | mif | oif |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in states:
            lines.append(
                f"| {s.turn} | {s.my_planets} | {s.op_planets} | {s.my_ships} | {s.op_ships} "
                f"| {s.my_prod} | {s.op_prod} | {s.my_in_flight} | {s.op_in_flight} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def enforce_size_budget(text: str, events: list[Event], render_fn) -> str:
    if len(text.splitlines()) <= MAX_LINES_BUDGET and len(text.encode("utf-8")) <= MAX_BYTES_BUDGET:
        return text
    trimmed_events = sorted(events, key=lambda e: e.turn)[:30]
    return render_fn(events_override=trimmed_events, include_full_override=False)


def process_one(
    *,
    replay_path: Path,
    label: str,
    out_path: Path,
    player_id: int | None,
    ship_loss_abs: int,
    ship_loss_rel: float,
    include_full: bool,
) -> None:
    replay = load_replay(replay_path)
    n_players = len(replay.get("rewards") or [])
    if n_players > 2:
        print(f"WARN: {replay_path.name} has {n_players} players (FFA). Treating others as aggregated opponent.", file=sys.stderr)
    pid = player_id if player_id is not None else infer_viewpoint(replay, label=label)
    states = extract_states(replay, pid)
    events = detect_events(replay, states, pid, ship_loss_abs, ship_loss_rel)

    def render(events_override=None, include_full_override=None):
        return render_markdown(
            label=label,
            replay_path=replay_path,
            replay=replay,
            pid=pid,
            states=states,
            events=events_override if events_override is not None else events,
            include_full=include_full_override if include_full_override is not None else include_full,
        )

    text = render()
    text = enforce_size_budget(text, events, render)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({len(text.splitlines())} lines, {len(text.encode('utf-8'))} bytes)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", required=True, type=Path, help="Replay .json.gz path (specify twice)")
    parser.add_argument("--label", action="append", required=True, type=str, help="Label for each replay (same order as --replay)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--ship-loss-abs", type=int, default=20)
    parser.add_argument("--ship-loss-rel", type=float, default=0.30)
    parser.add_argument("--full-stats", action="store_true")
    args = parser.parse_args()

    if len(args.replay) != 2 or len(args.label) != 2:
        parser.error("Exactly 2 --replay and 2 --label are required.")

    for idx, (replay_path, label) in enumerate(zip(args.replay, args.label, strict=True), start=1):
        out_path = args.out_dir / f"result_{idx}.md"
        process_one(
            replay_path=replay_path,
            label=label,
            out_path=out_path,
            player_id=args.player_id,
            ship_loss_abs=args.ship_loss_abs,
            ship_loss_rel=args.ship_loss_rel,
            include_full=args.full_stats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
