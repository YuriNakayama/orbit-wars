# rulebase/case3 JAX port — result

> 記録: 2026-06-03 ~14:15 / 状態: done (Step1+非劣化) / 親: ../20260603_case8_jax_port/

## case3 = case8-lineage + shallow rollout (ROLLOUT_ENABLED)

- 差別化機能 = **rollout** (top-K mission を true2p で敵 1-ply 応答想定し reorder)。
  これは active な機能 (dormant でない)。**但し launch 出力は滅多に変わらない**:
  case3 Python vs case8 Python = 7/150 turn mismatch (**5%**)。reorder しても top
  mission が大体不変のため。
- config は case2 と同じ case1 寄り値 (PARTIAL=6, REINFORCE 2/0.75, opening 0.9/0.86)。

## 実装 + 検証

- case8 core_jax を copy + case3 config 適用。**rollout は JAX に未 port** (nested
  simulation、config-only port)。結合テスト作成。
- foreground gate:
  - seeds 0-1: JAX 1/4 = 25% (seed1 js=0 が rollout 差で PY に flip)
  - seeds 2-3: JAX 4/4 = 100%
  - **通算 5/8 = 62.5%** → 25% は n=4 noise、非劣化は明確 (むしろ勝ち越し)。

## 限界 (honest)

rollout (5% の mission reorder) は **未 port = documented gap**。特定 seed では case3
Python に劣後するが ≈0勝劣化ではない (通算 62.5%)。nested-sim rollout の JAX 化は
コスト大 vs 効果 5% で見送り。

→ **case3 完了** (config-only port + 非劣化)。case2/3/4/8/9 が core 共有。
教訓: 4-game gate は n=4 で noise 大、border line 時は +追加 seed で確認。
