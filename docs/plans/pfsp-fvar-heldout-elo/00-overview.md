# f_var matchmaking + held-out eval + Elo — Overview

時刻: 2026-06-08 / branch: `feature/pfsp-fvar-heldout-elo` / 対象: `bot/pipeline/reinforce/case7/`

## 背景 (なぜこの feature か)
case7 の RL は JAX化 rulebase (`baseline_jax_full`) + 過去 self snapshot を相手に
PPO 自己対戦する。直近の実験ループ (別ブランチ H1-H4) で判明:
- opponent 難度調整 (H1)・dense報酬 (H2)・BC warm-start (H3) のいずれも
  vs rulebase 勝率 ~0.3 の天井を破れず、entropy collapse する。
- 共通因子は「相手分布が agent の実力から乖離している」こと。強すぎ相手では
  勾配が死に、弱すぎ相手では interaction を浪費する (FSP の既知欠点)。

## この feature の狙い
**相手の強さ分布を agent の現在実力の近傍 (勝率 ~0.5) に集中させる** matchmaking
(PFSP `f_var`) を導入し、密な学習信号を維持する。

ただしそうすると **対戦勝率は構造的に ~0.5 に固定され、進捗指標として使えなくなる**。
そこで進捗を **matchmaking 非依存の絶対量**で測る 3 点セットを追加:

| 指標 | 役割 | matchmaking 非依存か |
|---|---|---|
| **held-out 勝率** (固定相手・固定seed N iter毎) | 絶対進捗 | ✅ 固定相手なので影響なし |
| **agent Elo** (固定基準にアンカー) | 相対skill | ✅ 固定基準 |
| **policy entropy** | 健全性 (collapse 検出) | ✅ |
| (参考) match 勝率 | 学習信号の確認のみ (~0.5 期待) | ❌ 進捗には使わない |

## 構成
- [`01-design.md`](01-design.md) — 研究的根拠と matchmaking / 測定ロジックの設計
- [`02-results.md`](02-results.md) — H5 GPU run の結果記録

## 関連 (別ブランチの先行知見)
- `feature/reinforcement-learning-pooling-simple` の
  `docs/experiment/reinforce/20260608_case7_beat_rulebase/` (H1-H4 の棄却記録)
- `docs/plans/case7-pool-rl/07-research-pool-and-zero-winrate.md` (pool 構成 research)
