# Reinforce/case5 — support_reward (iter2) RESULT

> 関連: iter2_plan.md / hypotheses.md / iter1_result.md
> run_id: 20260527-182312__feature-support-reward__c359b68__seed0 / commit: c359b68 / case: reinforce_case5_kaggle_jax_train_h2_ratio
> 開始: 2026-05-27T18:23Z / 終了: 2026-05-27T22:04Z / wall-clock: 215.8 min / GPU: A100 80GB (SECURE) / コスト: ~$5.0

## Summary
H2 (保持割合差分 shaping, ratio mode, potential = mine/(mine+enemy) for ship & planet, coef=0.50) は
**強く支持された**。lite phase (iter 5–199) の last-10 win_rate = **0.763**、trend = **+0.651** で、
H1 combined (0.549 / +0.376) と planets baseline (~0.50) を **+21pp** 上回る。20-iter chunk が
0.179→…→0.741→0.767 と plateau なく単調増加。割合正規化 ([0,1]) により報酬スケールが安定し、
value_loss も 0.005 と極小。ratio が本 case の support reward として最有力。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | H2 ratio | H1 combined | planets baseline | note |
|---|---|---|---|---|
| last-10 win_rate (lite) | **0.763** | 0.549 | ~0.50 | **H1 比 +21pp** |
| trend (slope×n) | **+0.651** | +0.376 | +0.305 (case1 AA) | 収束大幅加速 |
| max win_rate (lite) | 0.813 | 0.641 | 0.609 | — |
| 20-chunk means (lite) | 0.179/0.265/0.352/0.551/0.568/0.592/0.66/0.713/0.741/0.767 | (頭打ち 0.55) | — | plateau なし |
| max approx_kl | 0.0047 | 0.0055 | — | trust region 健全 |
| value_loss (last-10) | **0.005** | 0.432 | — | ratio 正規化で激減 |
| iterations | 200 完走 | 200 完走 | — | exit_0 |
| runtime | 215.8 min (A100) | 135.5 min (4090) | — | A100 が本 workload で遅め |

`shaping_mode=ratio`, coef=0.50 を history 全行で確認。

## Diagnosis
- **割合正規化が決定打**。絶対数 (H1) は production スケールで値が膨らみ value head の回帰が荒れる
  (value_loss 0.43) のに対し、ratio は [0,1] に収まり value_loss 0.005 と桁違いに安定。
  安定した value 推定が advantage の質を上げ、policy 改善が加速した。
- ratio Δ は「保持割合が増えた/減った」を直接報酬化するため、序盤の領域取り合いで密な勾配が立ち、
  chunk means が H1 の頭打ち (0.55) を大きく超えて 0.77 まで伸びた。
- A100 で 215 分と長く ~$5 要したが、成果は明確。次の H4/H5/H3 は 3090/4090 限定で実施しコスト抑制。

## Decision
- 採否: **adopted (n<300 で確定保留だが +21pp は noise floor を大幅超過)** — ratio shaping は
  H1・baseline を圧倒。case5 の support reward の中核として採用筋。
- 次の一手: H4(係数 sweep) は ratio に対する coef 最適化として再定義する価値あり (現状は H1 前提)。
  まず H7 (ratio の clip/正規化派生) や H5 (production potential) と組み合わせる余地。
  promote は保留 (user 承認必須、+21pp なので promote 候補筆頭)。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train_h2_ratio/runs/20260527-182312__feature-support-reward__c359b68__seed0/best.pt`
- metrics: 同ディレクトリ `metrics.json`
- logs: train.log / onstart.log / gpu.log / system.log
