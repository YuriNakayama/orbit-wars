# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Game constants and tuning parameters ported verbatim from the notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BOARD: float = 100.0
CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
MAX_SPEED: float = 6.0
SUN_SAFETY: float = 1.5
ROTATION_LIMIT: float = 50.0
TOTAL_STEPS: int = 500
HORIZON: int = 110
LAUNCH_CLEARANCE: float = 0.1

EARLY_TURN_LIMIT: int = 40
OPENING_TURN_LIMIT: int = 80
LATE_REMAINING_TURNS: int = 60
VERY_LATE_REMAINING_TURNS: int = 25

SAFE_NEUTRAL_MARGIN: int = 2
CONTESTED_NEUTRAL_MARGIN: int = 2
INTERCEPT_TOLERANCE: int = 1

SAFE_OPENING_PROD_THRESHOLD: int = 4
SAFE_OPENING_TURN_LIMIT: int = 10
ROTATING_OPENING_MAX_TURNS: int = 13
ROTATING_OPENING_LOW_PROD: int = 2
FOUR_PLAYER_ROTATING_REACTION_GAP: int = 3
FOUR_PLAYER_ROTATING_SEND_RATIO: float = 0.62
FOUR_PLAYER_ROTATING_TURN_LIMIT: int = 10

COMET_MAX_CHASE_TURNS: int = 10

ATTACK_COST_TURN_WEIGHT: float = 0.55
SNIPE_COST_TURN_WEIGHT: float = 0.45
INDIRECT_VALUE_SCALE: float = 0.15
INDIRECT_FRIENDLY_WEIGHT: float = 0.35
INDIRECT_NEUTRAL_WEIGHT: float = 0.9
INDIRECT_ENEMY_WEIGHT: float = 1.25

STATIC_NEUTRAL_VALUE_MULT: float = 1.4
STATIC_HOSTILE_VALUE_MULT: float = 1.55
ROTATING_OPENING_VALUE_MULT: float = 0.95
HOSTILE_TARGET_VALUE_MULT: float = 1.85
OPENING_HOSTILE_TARGET_VALUE_MULT: float = 1.45
SAFE_NEUTRAL_VALUE_MULT: float = 1.2
CONTESTED_NEUTRAL_VALUE_MULT: float = 0.7
EARLY_NEUTRAL_VALUE_MULT: float = 1.2
COMET_VALUE_MULT: float = 0.65
SNIPE_VALUE_MULT: float = 1.12
SWARM_VALUE_MULT: float = 1.05
FINISHING_HOSTILE_VALUE_MULT: float = 1.15
BEHIND_ROTATING_NEUTRAL_VALUE_MULT: float = 0.92

NEUTRAL_MARGIN_BASE: int = 2
NEUTRAL_MARGIN_PROD_WEIGHT: int = 2
NEUTRAL_MARGIN_CAP: int = 8
HOSTILE_MARGIN_BASE: int = 3
HOSTILE_MARGIN_PROD_WEIGHT: int = 2
HOSTILE_MARGIN_CAP: int = 12
STATIC_TARGET_MARGIN: int = 4
CONTESTED_TARGET_MARGIN: int = 5
FOUR_PLAYER_TARGET_MARGIN: int = 3
LONG_TRAVEL_MARGIN_START: int = 18
LONG_TRAVEL_MARGIN_DIVISOR: int = 3
LONG_TRAVEL_MARGIN_CAP: int = 8
COMET_MARGIN_RELIEF: int = 6
FINISHING_HOSTILE_SEND_BONUS: int = 3

STATIC_TARGET_SCORE_MULT: float = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT: float = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT: float = 0.84
DENSE_STATIC_NEUTRAL_COUNT: int = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT: float = 0.90
SNIPE_SCORE_MULT: float = 1.12
SWARM_SCORE_MULT: float = 1.06

FOLLOWUP_MIN_SHIPS: int = 8
LOW_VALUE_COMET_PRODUCTION: int = 1
LATE_CAPTURE_BUFFER: int = 5
VERY_LATE_CAPTURE_BUFFER: int = 3

REAR_SOURCE_MIN_SHIPS: int = 16
REAR_DISTANCE_RATIO: float = 1.25
REAR_STAGE_PROGRESS: float = 0.78
REAR_SEND_RATIO_TWO_PLAYER: float = 0.62
REAR_SEND_RATIO_FOUR_PLAYER: float = 0.7
REAR_SEND_MIN_SHIPS: int = 10
REAR_MAX_TRAVEL_TURNS: int = 40

PARTIAL_SOURCE_MIN_SHIPS: int = 16
MULTI_SOURCE_TOP_K: int = 5
MULTI_SOURCE_ETA_TOLERANCE: int = 2
MULTI_SOURCE_PLAN_PENALTY: float = 0.85

SWARM_MIN_PARTICIPANT_SHIPS: int = 32
FULL_COMMIT_THRESHOLD_SHIPS: int = 40
FULL_COMMIT_FRACTION: float = 0.85

HOSTILE_SWARM_ETA_TOLERANCE: int = 1

THREE_SOURCE_SWARM_ENABLED: bool = True
THREE_SOURCE_MIN_TARGET_SHIPS: int = 20
THREE_SOURCE_ETA_TOLERANCE: int = 1
THREE_SOURCE_PLAN_PENALTY: float = 0.75

REINFORCE_ENABLED: bool = True
REINFORCE_MIN_PRODUCTION: int = 2
REINFORCE_MAX_TRAVEL_TURNS: int = 22
REINFORCE_SAFETY_MARGIN: int = 5
REINFORCE_VALUE_MULT: float = 1.35
REINFORCE_MAX_SOURCE_FRACTION: float = 1.0
REINFORCE_MIN_FUTURE_TURNS: int = 40

MULTI_ENEMY_PROACTIVE_HORIZON: int = 14
MULTI_ENEMY_PROACTIVE_RATIO: float = 0.22
MULTI_ENEMY_STACK_WINDOW: int = 3

PROACTIVE_DEFENSE_HORIZON: int = 12
PROACTIVE_DEFENSE_RATIO: float = 0.18

LATE_IMMEDIATE_SHIP_VALUE: float = 0.6
WEAK_ENEMY_THRESHOLD: int = 45
ELIMINATION_BONUS: float = 18.0

BEHIND_DOMINATION: float = -0.20
AHEAD_DOMINATION: float = 0.18
FINISHING_DOMINATION: float = 0.35
FINISHING_PROD_RATIO: float = 1.25
AHEAD_ATTACK_MARGIN_BONUS: float = 0.08
BEHIND_ATTACK_MARGIN_PENALTY: float = 0.05
FINISHING_ATTACK_MARGIN_BONUS: float = 0.08

DOOMED_EVAC_TURN_LIMIT: int = 24
DOOMED_MIN_SHIPS: int = 8

CRASH_EXPLOIT_ENABLED: bool = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS: int = 10
CRASH_EXPLOIT_ETA_WINDOW: int = 2
CRASH_EXPLOIT_POST_CRASH_DELAY: int = 1
CRASH_EXPLOIT_VALUE_MULT: float = 1.18

COMET_NPV_ENABLED: bool = False
COMET_NPV_MIN_USABLE_TURNS: int = 6
COMET_NPV_SHORTLIFE_FLOOR: float = 0.35
COMET_NPV_LONGLIFE_BONUS: float = 1.15
COMET_NPV_LONGLIFE_TURNS: int = 40

DYNAMIC_PROACTIVE_HORIZON_ENABLED: bool = False
DYNAMIC_PROACTIVE_HORIZON_MAX: int = 24
DYNAMIC_PROACTIVE_HORIZON_PROD_STEP: int = 3
DYNAMIC_PROACTIVE_HORIZON_PROD_CAP: int = 5

SAFE_INTERCEPT_HALF_STEP: bool = True

FINISHING_TIE_GUARD: bool = False

# --- STAY judge (case6 only) -------------------------------------------------
# When STAY decides to hold ships at a source, the per-source attack-budget is
# reduced by `held_ships` for the current turn. Holding ships keeps them
# available for next-turn defense (planet-resolved combat) or for next-turn
# burst-launch (higher fleet_speed when more ships are sent together).
STAY_ENABLED: bool = True
# Disabled per ablation vs baseline_v5 (iter2_result.md): defense-only 52% vs
# burst-only 59% — defense hold suppresses launches without delivering the
# expected fleet-peak gain. Burst-only is the production default.
STAY_DEFENSE_ENABLED: bool = False
STAY_DEFENSE_HORIZON: int = 12
# Risk × value weighting threshold. Lower → STAY fires for weaker threats.
STAY_DEFENSE_THRESHOLD: float = 1.0
# Sources too far from threatened planets cannot help within the horizon.
STAY_DEFENSE_MAX_TRAVEL_TURNS: int = 18
STAY_BURST_ENABLED: bool = True
# iter3 broad setting (gain>=1, ships>=8, dist<=30): 54.7% vs v5 (300 eps).
# iter4 tightened (2/12/20) collapsed to 41% — broad accumulation is the actual
# win driver, not per-hold quality.
# Required ETA improvement (turns) when comparing send-now vs send-next.
STAY_BURST_MIN_GAIN: int = 1
# Below this size the speed curve is too flat to benefit from accumulation.
STAY_BURST_MIN_SHIPS: int = 8
# Distant targets dilute the burst benefit; cap reasonable BURST candidates.
STAY_BURST_MAX_TARGET_TURNS: int = 30
# Maximum consecutive turns a single source may be held by burst. After this,
# the source is forced to launch (or whatever strategy.py decides) for at least
# one turn before it can be held again. iter4 found that broad burst's win
# contribution comes from cumulative pressure, but the same source holding for
# many turns lets context shift around it ("stuck holds"). Capping at 3 keeps
# the cumulative effect alive while preventing context-loss freezes.
STAY_BURST_MAX_HOLD_TURNS: int = 3

# --- ACCUMULATE mission (case7 only) ----------------------------------------
# 多ターン蓄積: 敵脅威スコアが低い友軍 source で「目標惑星捕獲必要量
# + safety + fleet_speed knee」までの ships 数が揃うまで複数ターン保持し、
# 揃った時点で遠距離 (ETA >= ACCUMULATE_MIN_TARGET_TURNS) の友軍 / 敵
# 惑星に単発攻撃する mission を発火させる。case6 STAY_BURST が 1 ターン
# arbitrage であるのに対し、ACCUMULATE は target-aware なしきい値達成
# まで複数ターン待つことで fleet_speed カーブの knee 以上を確実に踏む。
ACCUMULATE_ENABLED: bool = True
# 蓄積を許可する敵脅威スコア上限。reserve > 0 (= short-horizon の駐留
# 不足が予測される) の source は accumulate に回さず通常の defense /
# 通常 mission に譲る。
ACCUMULATE_THREAT_RESERVE_MAX: int = 0
# capture 必要量への安全上乗せ。target.ships の伸びと fleet 中の損失分。
ACCUMULATE_SAFETY_SHIPS: int = 4
# fleet_speed knee 近傍。MAX_SPEED=6 のとき log(60)/log(1000) ≈ 0.59、
# ratio^1.5 ≈ 0.45、speed ≈ 3.3 (ratio=1 の半分強)。これを下限に
# 取ることで knee 未満の小規模単発を弾く。
ACCUMULATE_KNEE_SHIPS: int = 60
# accumulate phase は近距離は STAY/通常 mission に任せる。ETA がこれ
# 以上の遠距離 target のみを対象にする。
ACCUMULATE_MIN_TARGET_TURNS: int = 15
# accumulate phase の ETA 上限。これより遠い target は orbit/comet で
# 大きく動くため、待機しても命中率が下がる。
ACCUMULATE_MAX_TARGET_TURNS: int = 60
# 同一 source が連続して accumulate hold できるターン数の上限。超過
# したら強制発火 (lift hold) する。case6 burst の MAX_HOLD_TURNS=3 と
# は独立に管理。case6 と違い「揃うまで」の蓄積なので長めに許容。
ACCUMULATE_MAX_HOLD_TURNS: int = 12
# accumulate score の係数。target の indirect_wealth と production を
# 重みに使い、他 mission との score 競合に乗せる。
ACCUMULATE_VALUE_MULT: float = 1.0
# accumulate fire score の調整: ETA が長いほど価値が薄れるので、
# turns で割る weight。
ACCUMULATE_COST_TURN_WEIGHT: float = 0.4
# case10 t14 罠対策: 序盤 step では accumulate 系 (hold + fire) を発動しない。
# memory: project_case7_t14_trap (10戦集計で trigger 確率 70% を実証)。
# 0 にすると case7 等価動作 (ガード無効)。
ACCUMULATE_MIN_LAUNCH_STEP: int = 30

# iter3: Phase 5 thrash 抑制
# 同 planet への capture/snipe/swarm を直近 N ターン以内に LIMIT 回以上
# commit していたら score を MULT 倍に減衰。replay 分析で見た
# t90-150 の planet thrash 連鎖 (planet#2 を 32T に 5 回 thrash) を構造的に抑制。
THRASH_REPEAT_FILTER_ENABLED: bool = (
    False  # iter3 result: filter 機能するも勝率不変、撤退
)

# iter4: accumulate threshold の need-based 動的化
# iter4 30戦結果: vs v4 = 40.0% (iter1/iter3 の 53.3% から -13pp 悪化)。
# Type A (早期消滅) は 8→6 に改善するも、Type B (長期消耗) が 6→12 に倍増。
# 動的化 (small-need で送信量削減) → 敵 60ships 反撃に押し負けて取り返される
# wear-out パターンが増殖。撤退、FLOOR=60 (= 旧 KNEE_SHIPS 等価) に戻す。
# `ACCUMULATE_KNEE_SHIPS=60` は probe_speed 計算用に維持。
ACCUMULATE_KNEE_SHIPS_FLOOR: int = 60
ACCUMULATE_KNEE_OVERSHOOT_RATIO: float = 1.5
THRASH_REPEAT_WINDOW: int = 8
THRASH_REPEAT_LIMIT: int = 2
THRASH_REPEAT_SCORE_MULT: float = 0.4

HARASS_ENABLED: bool = True
HARASS_MIN_TARGET_PRODUCTION: int = 2
HARASS_MIN_TARGET_SHIPS: int = 1
HARASS_MAX_TRAVEL_TURNS: int = 20
HARASS_MIN_SRC_RESERVE: int = 10
HARASS_COST_TURN_WEIGHT: float = 0.5
HARASS_VALUE_MULT: float = 1.0
HARASS_PRODUCTION_STEAL_TURNS: int = 5

OPPONENT_MODEL_ENABLED: bool = False
OM_HISTORY_WINDOW: int = 5
OM_MIN_LAUNCH_SHIPS: int = 3
OM_MIN_LAUNCH_STOCK: int = 10
OM_LAUNCH_SOURCE_BONUS: float = 0.0
OM_PREFERENCE_BONUS: float = 0.0
OM_PROACTIVE_THREAT_BOOST: float = 0.0
OM_PREDICT_HORIZON: int = 20
OM_MAX_HISTORY_ENTRIES: int = 600

OM_V2_ENABLED: bool = False
OM_PREDICTION_WEIGHT: float = 0.5
OM_PREDICT_MIN_LAUNCHES: int = 2
OM_PREDICT_APPLY_AFTER_STEP: int = 30
OM_DEFEND_PREDICTED_ATTACK: bool = True

LOOKAHEAD_ENABLED: bool = False
LOOKAHEAD_MAX_DEPTH: int = 1
LOOKAHEAD_APPLY_AFTER_STEP: int = 20
LOOKAHEAD_PREDICTION_WEIGHT: float = 0.6


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML copy of these constants for reference / future tuning."""
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"baseline config at {path} must be a YAML mapping")
    return data
