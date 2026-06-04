# rulebase/case2 JAX port — result

> 記録: 2026-06-03 ~14:00 / 状態: done (Step1+非劣化) / 親: ../20260603_case8_jax_port/

## case2 = 旧 case1-lineage fork (config が case1 寄り)

- case2 config は case8 と 18 行差だが **値は case1 寄り**: PARTIAL_SOURCE_MIN_SHIPS=6,
  REINFORCE_SAFETY_MARGIN=2, REINFORCE_MAX_SOURCE_FRACTION=0.75,
  ROTATING_OPENING_VALUE_MULT=0.9, DENSE_ROTATING_NEUTRAL_SCORE_MULT=0.86。
- → case8 core_jax を copy し **case2 の config 値に revert** (5 定数)。harass 等の
  mission 構造は共通。

## 実装 + 検証

- case8 baseline_jax/core_jax を case2 へ copy + config 値適用。
- 結合テスト作成、**foreground 4-game gate: JAX 2/4 = 50%** (非劣化)。import clean。

→ **case2 完了**。case2/4/8/9 が core 共有 (config 値のみ差)。
残 lineage: case3 (rollout.py 追加), case6/7 (stay.py 追加)。case5 別 shape。
