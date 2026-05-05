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

# ----- case8 multi-step beam search optimizer -----
# iter1 (32.3%) / iter2 (27.0%) ともに vs v4 で大幅未達 → 撤退。
# default を False にし、case8 は case7 等価な greedy 経路を踏む。
# 改修して再評価する際は True に戻す。
BEAM_ENABLED: bool = False
# 何ターン先まで simulate するか (1=ターン内のみ, 2-3=多ターン展開)。
BEAM_HORIZON: int = 2
# 各 depth で残す上位 plan 数。
BEAM_WIDTH: int = 4
# 1 plan あたりに展開する分岐数 (top-N 候補ミッションから採用 / drop / no-op)。
BEAM_BRANCH_LIMIT: int = 8
# 敵モデル: "static" は予測 arrivals 据え置き、"true2p_light" は
# 敵 plan_moves(light) 反応を folding。"true2p_sampled" は
# `step % BEAM_OPPONENT_SAMPLE_STRIDE == 0` の turn だけ folding し、
# 平均計算負荷を下げる (iter2 default、smoke で turn_p95 ≤ 0.4s 想定)。
BEAM_OPPONENT_MODE: str = "true2p_sampled"
# true2p_sampled モードで何 turn ごとに敵反応を folding するか。
BEAM_OPPONENT_SAMPLE_STRIDE: int = 5
# 評価関数 mode: "mission_score" は採用 missions の mission.score 合計
# (iter2 default、greedy と整合)、"legacy_net_ships" は iter1 の
# horizon-end net_ships 線形和 (劣化確認済み)。
BEAM_EVALUATOR_MODE: str = "mission_score"
# 評価関数の重み (自軍純艦数, 自軍 production 合計, 敵 home 脅威スコア)。
# legacy mode のみ参照。
BEAM_VALUE_WEIGHTS: tuple[float, float, float] = (1.0, 0.5, 0.3)
# 1 ターンあたりの beam 探索時間上限 (秒)。超過時は best-so-far で early stop。
BEAM_TIME_BUDGET_S: float = 0.6

# ----- iter3: planet thrash filter -----
# 直近 THRASH_WINDOW ターン以内に自軍が奪われた / 自軍が同 planet 宛 mission を
# THRASH_REPEAT_COMMIT_LIMIT 回以上 commit した planet への
# capture/snipe/swarm mission の score を THRASH_SCORE_MULT 倍に減衰。
# replay 分析 (20260505_1410_iter2) で確認した planet thrash 連鎖を構造的に抑制。
THRASH_FILTER_ENABLED: bool = True
THRASH_WINDOW: int = 10
THRASH_SCORE_MULT: float = 0.3
# iter3 v0 では `_record_mission_commits` が move 種別を区別せず
# (harass / accumulate / followup 等を含めて) 記録していたため、
# `mission_commits >= 2` 条件が常に真になり filter が全 planet で誤発火。
# v1 では mission_commits 経路を無効化 (limit=999)、recently_lost 単独で判定。
# 改めて mission 種別を絞った record 実装ができたら 2 に戻す。
THRASH_REPEAT_COMMIT_LIMIT: int = 999


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML copy of these constants for reference / future tuning."""
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"baseline config at {path} must be a YAML mapping")
    return data
