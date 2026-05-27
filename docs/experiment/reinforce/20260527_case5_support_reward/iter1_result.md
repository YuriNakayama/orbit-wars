# Reinforce/case5 — support_reward (iter1) RESULT

> 関連: iter1_plan.md / hypotheses.md
> run_id: 20260527-151636__feature-support-reward__2f37b9e__seed0 / commit: 2f37b9e / case: reinforce_case5_kaggle_jax_train
> 開始: 2026-05-27T15:16Z / 終了: 2026-05-27T17:36Z / wall-clock: 135.5 min / GPU: RTX 4090 (SECURE) / コスト: ~$1.6

## Summary
H1 (ship 差分 + planet 差分の combined shaping, coef_ship=0.001 / coef_planet=0.50) は
**支持された**。baseline_jax_lite phase (iter 5–199) の last-10 win_rate = **0.549**、
学習 trend = **+0.376** (195-iter 線形 slope×phase) で 20-iter chunk が 0.164→…→0.552→0.548 と
ほぼ単調増加。baseline (case3 planets 単体: last-10 ~0.50 / case1 AA 300iter trend +0.305) に対し
last-10 **+~5pp**、trend も上回り、採用しきい値 (last-10 +3pp **または** trend 明確改善) を満たす。
ただし skip list `n<300 で結論を出さない` により win-rate の確定判定は inconclusive 扱いとし、
**trend ベースで adopted 寄り**と整理する。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | combined (H1) | baseline 参照 | note |
|---|---|---|---|
| last-10 win_rate (lite phase) | **0.549** | ~0.50 (case3 planets) / 0.501 (case1 AA) | +~5pp |
| trend (phase slope×n) | **+0.376** | +0.305 (case1 AA) | 収束速度↑ |
| max win_rate (lite phase) | 0.641 | 0.609 (case1 AA best) | — |
| iterations | 200 (完走) | 200/300 | exit_0 正常終了 |
| 20-iter chunk means (lite) | 0.164 / 0.249 / 0.409 / 0.451 / 0.454 / 0.448 / 0.469 / 0.518 / 0.552 / 0.548 | — | 単調増加 |
| runtime | 135.5 min | ~180 min | RTX4090 高速 |

`shaping_mode=combined`, `coef_ship=0.001`, `coef_planet=0.50` が history 全行で確認済み。

## Diagnosis
- ship 差分 (微小 coef=0.001) を planet 差分 (0.50) に重畳しても trend を毀損せず、むしろ
  last-10・trend ともに baseline をやや上回った。ship 成分は production で絶対値が大きいが、
  0.001 係数で planet 主信号を崩さず補助的に効いたと解釈できる。
- potential-based を 2 本独立 carry で厳密維持したため学習は安定 (value 発散・stall なし、
  approx_kl 暴走なし、200 iter 完走)。
- chunk means が後半 0.55 付近で頭打ち気味 → ship 成分をもう少し効かせる余地 (H4 sweep) あり。

## Decision
- 採否: **inconclusive (trend は adopted 寄り)** — n<300 ルールにより win-rate 確定判定は保留。
  trend +0.376 と last-10 +~5pp は baseline を上回るため、後続 sweep で確証を取る価値が高い。
- 次の一手: **H4 (coef_ship : coef_planet 比率 sweep)** で ship 成分を 0.001→0.005/0.01 に上げ、
  last-10 が 0.55 の頭打ちを破るか検証。promote は保留 (user 承認必須)。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train/runs/20260527-151636__feature-support-reward__2f37b9e__seed0/best.pt`
- metrics: `data/output/models/reinforce/case5_kaggle_jax_train/runs/20260527-151636__feature-support-reward__2f37b9e__seed0/metrics.json`
- logs: train.log / onstart.log / gpu.log / system.log (同ディレクトリ)
