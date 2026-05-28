# Reinforce/case5 — support_reward (iter4) RESULT

> 関連: iter4_plan.md / hypotheses.md / iter3_result.md
> run_id: 20260528-050116__feature-support-reward__531f725__seed0 / commit: 531f725 / case: reinforce_case5_kaggle_jax_train_h5_ratio_prod
> 開始: 2026-05-28T05:01Z / 終了: 2026-05-28T07:14Z / wall-clock: 128.3 min / GPU: RTX 4090 (SECURE) / コスト: ~$1.5

## Summary
H5 (production 加重保持割合 ratio_prod, coef=1.0) は **支持されなかった (rejected)**。lite phase
last-10 = **0.771** で、現行最良 H4 (count-based ratio, 0.820) を **−4.9pp** 下回った。trend も
+0.601 (< H4 +0.668)、20-chunk が後半 0.797→0.781 と軽い後退を見せ plateau 兆候。
planet を production 加重にすると高 prod 惑星 (home 等) に報酬が偏り、序盤の neutral 確保や
領域の「広さ」確保を軽視する副作用が出たと解釈できる。**count ベース (H4) が最良のまま維持**。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | H5 ratio_prod | H4 ratio (count, best) | note |
|---|---|---|---|
| last-10 win_rate (lite) | 0.771 | **0.820** | **−4.9pp (劣後)** |
| trend (slope×n) | +0.601 | +0.668 | やや低下 |
| max win_rate (lite) | 0.867 | 0.906 | −3.9pp |
| 20-chunk means (lite) | 0.226/0.36/0.496/0.598/0.639/0.679/0.713/0.782/0.797/0.781 | (0.83 まで上昇) | 後半 0.797→0.781 後退 |
| max approx_kl | 0.0055 | 0.005 | 健全 |
| value_loss (last-10) | 0.0092 | 0.0066 | やや上昇 |
| iterations | 200 完走 | 200 完走 | exit_0 |
| runtime | 128.3 min (4090) | 126.5 min (4090) | 同等 |

## Diagnosis
- production 加重は「どの惑星を持つか」の質を報酬化する狙いだったが、Orbit Wars では home planet
  などごく少数の高 prod 惑星に重みが集中し、**惑星「数」(広さ) を確保するインセンティブが薄れた**
  と考えられる。count ベース (各惑星 equal) の方が序盤の領域拡大を素直に促し、結果として勝率が高い。
- value_loss が 0.0066→0.0092 と微増したのも、production 値の分布が広く ratio 後も signal が
  荒れたため。count ベースの [0,1] 一様性が学習安定に効いていたと整理できる。
- 構造的劣化 (planning 段階で予見した副作用が顕在化) のため、production 系は deepen しない。

## Decision
- 採否: **rejected** — production 加重は count ベース (H4) に −4.9pp 劣後。Orbit Wars では
  惑星の「数」確保が「質」より重要という知見。H4 (ratio count, coef=1.0) が引き続き現行最良。
- 次の一手: H5 系は打ち切り。残る H7 (ratio clip) は H4 で value 既に安定のため効果薄の見込み、
  H3 (絶対保持数 対照) / H6 (time bonus) を消化。promote 候補は依然 H4 (要承認)。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train_h5_ratio_prod/runs/20260528-050116__feature-support-reward__531f725__seed0/best.pt`
- metrics: 同ディレクトリ `metrics.json`
