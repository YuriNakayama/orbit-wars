# Reinforce/case5 — support_reward (iter6) RESULT

> 関連: iter6_plan.md / hypotheses.md / iter3_result.md (H4 ratio coef=1.0, 0.820 = 現行最良)
> run_id: 20260531-021927__feature-support-reward__dc1fa41__seed0 / commit: dc1fa41 / case: reinforce_case5_kaggle_jax_train_h3_dense
> 開始: 2026-05-31T02:19Z / 終了: 2026-05-31T04:58Z / wall-clock: 155 min / GPU: NVIDIA GeForce RTX 4090 (SECURE) / コスト: ~$1.83

## Summary
H3 (絶対保持数の非差分 dense 加算、`coef · mine_count` 毎 turn) は **rejected**。
lite phase last-10 = **1.0000** (見かけ +18pp) だが、これは **見かけ上の win-rate inflation**
で実力向上ではない。value_loss = **7163** (H4 0.0066、~1M× 巨大化)、報酬分布が dense 項で
発散し value head が学習不能。Ng et al. 1999 が警告する **非 PBRS 加算の最適方策バイアス**
(貯め込み・引き伸ばし) が完全に顕在化した教科書通りの結果。**PBRS の必要性を実証する対照群
として成功**。H3 系は deepen しない (hypotheses.md 例外条件適用)、H4 (PBRS) が現行最良維持。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | H3 dense (coef_ship=0.01, coef_planet=0.1) | H4 ratio (best) | note |
|---|---|---|---|
| last-10 win_rate (lite) | **1.0000** | 0.820 | +18pp (見かけ・inflation) |
| trend (slope×n) | +0.0000 | +0.668 | 飽和 (序盤から 1.0) |
| max win_rate (lite) | 1.0000 | 0.906 | inflation 上限 |
| 20-iter chunk means (lite) | 1.0/1.0/1.0/1.0/1.0/1.0/1.0/1.0/1.0/1.0/1.0 | (H4 序盤 0.26→終盤 0.83) | 序盤から全勝表示 |
| **last-10 value_loss** | **7162.96** | **0.0066** | ⚠️ **~1M× 巨大、value 学習破綻** |
| last-10 approx_kl | 0.00026 | — | KL は小 (policy 動かず) |
| max approx_kl (lite) | 0.02094 | 0.005 | H4 比 4× |
| iterations | 200 完走 | 200 完走 | = |
| runtime | 155 min (4090) | 126 min (4090) | 同等 |

## Diagnosis
- **win_rate inflation の原因**: `win_rate` は `outcome > 0` で判定される。dense 加算
  `coef · mine_count` は常に正値 (mine_count ≥ 0)、horizon=500 turns × ~30 mine_ships
  × 0.01 = 約150 + planets 加算で **cumulative reward が +1 terminal を完全に上書き**。
  実際の試合に負けても cumulative > 0 になり win_rate=1.0 と誤計測される。
- **value_loss=7163 の意味**: 報酬の分布が dense 加算で発散 (H4 では shaping ∈ [-1,1] 程度、
  H3 では数百〜数千)、value head が GAE returns に追従できない。policy gradient 計算が
  ノイズだらけになるが、報酬最大化方向が「mine_count を維持すること」になり、policy は
  「動かない (発射しない)」方向に学習。approx_kl 0.0003 でほぼ更新なし。
- **これは Ng 1999 の典型例**: 非 potential-based な dense 加算は最適方策をバイアス。
  実環境の「勝つ」目的と、reward 目的「mine_count 最大化」が乖離。事前見込み通り。

## Decision
- 採否: **rejected (PBRS必要性を実証)** — hypotheses.md 例外条件「H3 は明確に劣化なら
  deepen しない」を適用。win_rate inflation は劣化以上の構造的破綻、PBRS 系 (H2/H4) が
  最適方策を保つことの強い証拠。promote 候補は引き続き H4 (0.820)。
- 次の一手: H3 系 deepen なし。残り H6 (time bonus, P3, H3 と pair) を消化。
  H6 も非 PBRS だが H3 と異なり時間方向のシェイピングなので別問題として実施。
  H4 が現行最良で確定の方向、promote はユーザー承認後。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train_h3_dense/runs/20260531-021927__feature-support-reward__dc1fa41__seed0/best.pt` (注: weights は最適方策でない、参考用)
- metrics: 同ディレクトリ `metrics.json` (history 200 iter 全)
- 本 run の best.pt は **採用候補外** (win_rate inflation のため意味のあるモデルではない)。
