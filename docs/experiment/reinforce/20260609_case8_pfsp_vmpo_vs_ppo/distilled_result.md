# RESULT — rulebase 蒸留クローン × V-MPO は実 rulebase への勝率を動かすか

> 関連: poc_bc_result.md / vmpo_distilled.yaml / il_case9_rulebase.yaml
> V-MPO run: 20260610-165630__feature-poc-v-mpo__1aff880__seed0 (reinforce_case8_vmpo_distilled)
> BC 蒸留 run: 20260610-155129__feature-poc-v-mpo__680040f__seed0 (case9_rulebase) / commit: 1aff8803
> 実行: 2026-06-11 / GPU 計 ~55分 (BC 41ep ≈ 48分 + V-MPO 50it ≈ 25分) ≈ $0.85 / データ生成・評価: CPU 無料
> 成功基準: V-MPO 学習中/前後で実 baseline_v1 への勝率が向上すること

## Summary

**不支持 (rejected)。** PoC (Kaggle 棋譜 BC, flat 0%) の対策として「教師と相手の両方を
rulebase 化」する蒸留アプローチを実装・実行したが、**実 baseline_v1 への勝率軌跡は
全 4 点 0/24 で横ばい**のまま。配線 (distilled opponent / bc_warmstart / held-out) は
全て正常稼働し、学習中はクローンへの held-out が 0.106→0.156 と微改善したが、
**実 rulebase へは転移しなかった**。律速は蒸留ギャップ: 52.5% の exact 一致率の
クローンは決定論的な本人に 0/24 で全敗し、「本物の強さ」を学習ループに運べていない。

## Numbers

### 実 baseline_v1 (Python rulebase) との offline 対戦 — 各 24 戦

| 評価点 | PoC arm (Kaggle BC) | 蒸留 arm (rulebase clone) |
|---|---|---|
| 教師 BC 単体 | 0/24 | **0/24** ← 律速 (蒸留ギャップ) |
| ckpt_i000 (≈教師) | 0/24 | 0/24 |
| ckpt_i015 | 0/24 | 0/24 |
| ckpt_i030 | 1/24 (noise) | 0/24 |
| ckpt_i049 (最終) | 0/24 | 0/24 |

### 学習中の固定相手曲線・蒸留品質

- held-out (vs 蒸留クローン, 毎 iter 32ep): **0.106 → 0.156** (iter12 以降プラトー)
- 蒸留 BC 品質 (early-stop ep41, best ep31): val_target_acc 89.2% / noop_acc 97.5% /
  **fire_acc (発射先 exact) 52.5%** / fire ROC-AUC 0.975
- 学習配線の健全性: BC 133/133 ロード、distilled opponent 133/133 ロード、
  iter0 win vs noop 0.78-0.94 (PoC の 0.25 より大幅に活発な初期方策)

## Diagnosis

1. **蒸留ギャップが二重の壁**: (a) クローン自身が本人に 0/24 — 教師の ~半分の
   行動しか exact 再現できず、決定論的 rulebase はその逸脱を正確に咎める。
   (b) 学習相手としても「本物の覆し方」を教えられない — クローンに勝てるように
   なっても (0.16 まで微上昇) 実 v1 には 0/24 のまま。
2. **train/eval ギャップの再確認**: 「クローン相手の改善が本人に転移しない」は
   既知の「in-JAX 近似相手での改善が実相手に転移しない」と同型のより強い証拠。
3. 50 iter という短い RL も一因だが、held-out が iter12 でプラトーしており、
   単純な延長で破れる壁ではない可能性が高い。

## Decision

- 採否: **rejected** (蒸留クローン経由では rulebase への勝率は動かなかった)
- 次の一手 (優先順):
  1. **クローン品質の改善**: fire_acc 52.5% → 70-80% 目標。データ増 (1,000→3,000+
     試合、v1 vs v2/v3 の異種対戦でカバレッジ拡大)、DAgger 式 (クローンの訪問状態で
     教師に再ラベル)、ships 回帰の精緻化。
  2. **per-turn 28 秒を受け入れた低頻度 strict 相手**: rollout の一部 episode のみ
     strict_v1 と対戦 (1 iter に数十 turn だけ) — 本物の勾配を少量混ぜる。
  3. **PPO arm での同条件比較** (V-MPO 固有の問題でないことの確認)。
  4. 報酬設計の再検討: 勝敗 + ratio shaping では「本人に勝つ」信号が遠すぎる可能性。
     Generals.io 式 potential (planets/ships/production の log 比) の係数調整。

## Artifacts

- V-MPO run dir: `data/output/models/reinforce/case8_vmpo_distilled/runs/20260610-165630__feature-poc-v-mpo__1aff880__seed0/` (ckpt_i{000,015,030,049}.pt / metrics.json)
- 蒸留クローン: `data/output/models/imitation/case9_rulebase/runs/20260610-155129__feature-poc-v-mpo__680040f__seed0/best.pt` (DVC push 済)
- 蒸留 mart: `data/mart/imitation/case9_rulebase/{train,val}.parquet` (633k/74k frames, DVC push 済)
- 評価 JSON: `data/output/experiment/case8_distilled/{clone_vs_v1,traj_i000,traj_i015,traj_i030,traj_i049}.json`
