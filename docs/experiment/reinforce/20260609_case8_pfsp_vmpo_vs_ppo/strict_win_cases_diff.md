# strict勝利ケースと敗北の差分 (2026-06-17)

> 問い: strict_v1 held-out で勝てた17ケース (max 3/64) と、ほぼ全敗のモデルの差分は?

## 方法
held-out は固定seed(777000)で毎回同じ64試合。全 ladder run の held-out 評価点を
strict勝率帯で層別し、同時の full held-out (汎用地力) と reward設定を比較。
(per-seed勝敗の直接診断はCPU strict対戦が ~50分超で intractable、metrics層別で代替)

## 結果: 差分は「汎用地力 (full held-out)」

| strict勝率帯 | n | full mean | full max |
|---|---|---|---|
| **≥2/64 (峰)** | 3 | **0.812** | 0.844 |
| 1/64 | 14 | 0.771 | 0.828 |
| **0/64** | 84 | **0.632** | 0.875 |

- strict に勝てるモデルは **full が明確に高い (0.81 vs 0.63)**。strict勝率と地力は
  単調相関 — 「strict を倒す特殊技」でなく **総合的な強さの裾が strict にも僅かに届く**。
- 峰 (2-3/64) の3ケース (ladder21 iter30/40, ladder22 iter30) は全て
  **terminal_scale=1.0 / shaping=clean** = 素直な reward 設定。boosted/実験的 reward
  (terminal_scale=8 の ladder13, dense boost の ladder23/24) ではない。
- 注: full=0.875 が strict=0 帯にも在る (ladder24 iter0) → 地力が高くても strict勝利は
  確率的で n=64 では出ないこともある。つまり「地力が高い」は strict勝利の**必要条件
  だが十分でない**。

## 結論
**勝利ケースと敗北の差分 = 汎用地力 (full) の高さ + 素直な reward。**
- strict の 1-3/64 勝利は「特定の楽な盤面を引いた」のではなく、**地力の高いモデルが
  確率的に拾った勝ち**。固定seedだが勝つseedは地力で増える (0→1-3/64)。
- これは ladder21_strict_win_deepdive.md の「弱体化strict全段の訓練win一様底上げ
  (no_op_bias↓ + aim修正の地力)」の裾が held-out strict に漏れた、という所見と整合。
- 含意: **strict勝率を上げる最短は「地力 (full) をさらに上げる」こと** (no_op_bias,
  aim, 学習量)。ただし地力だけでは確率的に 0-5% 止まりで、**断崖を質的に超えるには
  序盤戦略の直接教示 (strict-BC) が必要** という campaign 総括は不変。

## Artifacts
- 全 ladder run metrics (held-out strict_v1 / full の層別)
- 関連: campaign_summary_ladder1-24.md, ladder21_strict_win_deepdive.md
