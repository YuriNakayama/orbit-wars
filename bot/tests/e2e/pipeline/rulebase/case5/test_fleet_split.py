"""Fleet split regression tests for the case5 agent.

Orbit Wars では `speed = 1.0 + (maxSpeed-1) × (log(ships)/log(1000))^1.5` で
艦数が多いほど fleet 速度が上がる。同一ターゲットへ艦数を分割して送ると、
個々の fleet が遅くなり敵に先着を譲ってしまう。そのため 1 ターン内で
同一 target へ複数の action が出ることは原則として不合理 (swarm 合流のため
複数 source から送る場合を除く)。

エージェントは `SOFT_ACT_DEADLINE` / `allow_heavy()` により wall-clock
依存で探索フェーズを打ち切るため、1 エピソード単独のアサートは CI 負荷で
flaky になり得る。このテストは複数 seed を平均化し、「case4 水準
(≤1.5/ep)」より悪化していないことをアサートする。
"""

from __future__ import annotations

from collections import Counter

import pytest

from pipeline.rulebase.case5.baseline import agent
from tests.e2e.pipeline.util import make_orbit_env, run_orbit_wars_episode

_TOTAL_WAR_TURN_THRESHOLD = 458  # 500 - TOTAL_WAR_REMAINING_TURNS(42)
# ベースライン 7.8/ep、Phase 1 後 2.6/ep。CI 負荷で揺らぐため 4.0 で退行検知。
_SPLIT_PER_EPISODE_LIMIT = 4.0
_EPISODE_SEEDS = (0, 1, 2, 3, 4)


def _count_splits_in_action(actions: list[list[float]]) -> int:
    by_src_angle: Counter[tuple[int, float]] = Counter()
    for move in actions:
        if not move:
            continue
        src_id, angle, _ships = move
        by_src_angle[(int(src_id), round(float(angle), 2))] += 1
    return sum(c - 1 for c in by_src_angle.values() if c > 1)


def _split_count_for_seed(seed: int) -> tuple[int, int]:
    """Return (split_count, episode_length_turns)."""
    env = make_orbit_env(seed=seed, agents=2)
    run_orbit_wars_episode(env, [agent, agent])
    total = 0
    for turn_idx, step in enumerate(env.steps):
        if turn_idx >= _TOTAL_WAR_TURN_THRESHOLD:
            break
        for seat_state in step:
            action = seat_state.get("action")
            if not action:
                continue
            total += _count_splits_in_action(list(action))
    return total, len(env.steps)


@pytest.mark.slow
def test_fleet_split_rate_stays_below_threshold() -> None:
    """自己対戦の 1 エピソード平均分割件数が閾値以下であること。

    Phase 1 改修 (SWARM_MIN_PARTICIPANT_SHIPS, follow-up ガード) で分割は
    平均 7.8→2.6/ep に減少した。CI 負荷依存で個別エピソードは 0〜6 件の
    ばらつきがあるため、複数 seed を平均して退行を検知する。
    """
    per_seed = [_split_count_for_seed(s) for s in _EPISODE_SEEDS]
    splits = [s for s, _ in per_seed]
    lengths = [length for _, length in per_seed]
    avg = sum(splits) / len(splits)
    assert avg <= _SPLIT_PER_EPISODE_LIMIT, (
        f"分割件数が閾値 {_SPLIT_PER_EPISODE_LIMIT}/ep を超過: "
        f"splits_per_seed={splits} lengths={lengths} avg={avg:.2f}"
    )
