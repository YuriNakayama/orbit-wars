# case10 — case4 大改造 (capture 強化 + sniper/kamikaze 多用)

case4 (baseline_v4 = production) のフルコピー上に、capture mission の score
weight 強化と sniper/kamikaze 多用化を試す派生 case。

## 採用戦略 (iter1)

case4 (baseline_v4, fleet_consolidation 入り) を起点に config.py の以下を変更:

1. **Capture 強化**:
   - `STATIC_NEUTRAL_VALUE_MULT 1.4 → 1.6` (静的 neutral の優先度を上げ、production
     高い惑星を確実に確保)
   - `EARLY_NEUTRAL_VALUE_MULT 1.2 → 1.4`
2. **Sniper / Kamikaze 多用**:
   - `HARASS_MIN_SRC_RESERVE 10 → 6` (低 ship reserve でも harass を発火)
   - `HARASS_PRODUCTION_STEAL_TURNS 5 → 8` (奪取後の保持期待値を上げ harass score 上昇)
   - `SNIPE_VALUE_MULT 1.12 → 1.30`

これらは memory `project_heuristic_search_saturation` (case4 base 上の探索は 53%
で飽和) を踏まえた「heuristic 系で 53% を超えるか試す」探索。

## 仮説

case9 anti-ping-pong は cooldown 系で +5pp 不可と確定 (9 iter 失敗)。
case10 では **score weight の調整 + 攻撃寄り設定** で異なる状態空間を試行し、
production 増産速度で v4 に並ぶことを狙う。

詳細: `docs/experiment/rulebase/20260506_case10_capture_kamikaze/iter1_plan.md`

## 成績

- vs baseline_v4: TBD (`evaluation/compare_v4.py` で測定)

## 構造

case4 と同型。違いは config.py の score weight 群のみ (iter1 時点)。
