# reinforce/case1 — ppo_stabilization (iter1) RESULT

> 関連: iter1_plan.md / hypotheses.md
> run_id: `20260518-013025__feature-reinforce-learning-case0__f406f85__seed0` / commit: f406f85 / case: reinforce_case1
> 開始: 2026-05-18T01:30:30Z / 終了: 2026-05-18T08:40:40Z / 実行時間: 25 304s (~7h) / コスト: ~$1.75 (cap $1.5 → user 承認で引き上げ)

## Summary

H1 (`target_kl=0.05` early stopping) を `ppo_update` epoch ループに導入し 100 iter × 16 ep を完走。**仮説は部分支持 (inconclusive)**。target_kl による次 epoch skip は機能 (epochs_run mean=1.43 / 2、57 iter で early stop 発火) し、PPO update の累積量は抑制された。一方で **minibatch 単位の approx_kl は依然として頻繁に target を超過** (max=1.57、27/100 iter で > 0.1) し、当初の primary metric `max approx_kl < 0.1` は未達成。bc_kl は std=0.14 と iter0 baseline (cpu_stable_v1 系 0.18–0.49) より低く、安定化の方向には進んだ。次手は H2 (lr 1e-4 → 3e-5) で per-update Δ を縮小し minibatch スパイクを根本抑制する方針が妥当。

## Numbers

skip list に従い対戦評価 (300 ep) は実施せず、**学習ログ (approx_kl / bc_kl / epochs_run / win_rate) のみで採否判定**。

### 主要メトリクス (採否判定対象)

| metric | value | iter1 plan 閾値 | 判定 |
|---|---|---|---|
| max approx_kl | **1.566** | < 0.1 | **NG** (大幅超過、`target_kl=0.05` 設定下でも minibatch スパイクが残る) |
| approx_kl > 0.1 iter 数 | **27 / 100** | — | trust region 守り切れず |
| approx_kl > 0.05 iter 数 | 94 / 100 | — | target 自体を minibatch 平均で破る iter が大多数 |
| approx_kl mean | 0.132 | — | iter0 cpu_stable_v1 (0.08–1.18) と同レベル |
| approx_kl std | 0.184 | — | スパイク特性は変わらず |
| bc_kl mean | 0.320 | — | — |
| bc_kl std | **0.136** | iter0 比 < 0.5× | **△** (iter0 本番 100 iter の std 未確定。cpu_stable_v1 の 0.31 比なら 0.44× で達成) |

### 補助メトリクス

| metric | value | 期待 | 判定 |
|---|---|---|---|
| epochs_run mean | **1.43 / 2.0** | < 2.0 (early stop 発火) | **OK** (early stop が想定通り 57 iter で発火) |
| epochs_run < 1.5 (= 1 epoch 完走で停止) | 57 / 100 iter | — | hard trust region 機構は機能 |
| clip_fraction mean | 0.328 | < 0.30 | △ (僅か超過、許容範囲) |
| best_win_rate (vs baseline_v1, 16 ep) | 0.125 | — | n=16 過小、参考値のみ |
| final_win_rate | 0.0625 | — | 改善方向だが n=16 で結論不可 |
| runtime | 7h (A4500) | ~1h plan | **超過** (A4500 ペース過小評価。L4 等価で 3-4h 想定) |

## Diagnosis

**機能した点**:
- `target_kl=0.05` による epoch 中断は **設計通り動作** (epochs_run mean 1.43、57 iter で 2 epoch 目を skip)。CleanRL / SB3 標準実装の効果通り、PPO update の累積量は抑制。
- bc_kl の std が cpu_stable_v1 比で約半減 (0.31→0.14) し、学習中の policy ドリフト変動は緩和方向。

**機能しなかった点**:
- **minibatch 単位の approx_kl スパイクは抑制できない**。`target_kl` の判定は epoch 完走後の minibatch 平均で行うため、epoch 内の単一 minibatch が approx_kl=1.5+ を出しても当該 update はそのまま走り、次 epoch を抑止するだけ。
- iter 後半でも max approx_kl が高水準 (last 10 iter で 0.054–0.318) で、 学習進行に伴う自然減衰は観測されず。

**根本原因の仮説**:
- per-update の weight Δ が大き過ぎる (lr=1e-4 が PPO + BC warm-start には強い)。H2 (lr 3e-5) で root cause に対処予定。
- BC warm-start により policy が saturated な点から出発 → 初期の advantage signal が極端値を取りやすい。H3 (kl_beta 0.5→0.1) で BC anchor を緩める案も独立して評価する余地。

## Decision

- **採否**: **inconclusive**
  - (a) `max approx_kl < 0.1` 不達成 → 単独では rejected 相当
  - (b) `bc_kl std` 改善 (iter0 cpu 系比 0.44×) → 部分的に支持
  - epochs_run mean=1.43 で機構自体は機能、root cause (lr per-update Δ) が未対処
- **採用扱い**: target_kl=0.05 設定は **継続** (副作用なし、機構として有効、コードは保持)。
- **次の一手**: H2 (lr 1e-4 → 3e-5) を H1 と stacking した状態で評価。per-update Δ 縮小で minibatch スパイクを直接抑制し、(a) 達成を狙う。

## Artifacts

- run_dir: `data/output/models/reinforce/case1/runs/20260518-013025__feature-reinforce-learning-case0__f406f85__seed0/`
- model: `best.pt` (13M, iter1 中の best win_rate=0.125 snapshot)
- metrics: `metrics.json` (100 iter history)
- logs: `train.log` / `onstart.log` / `gpu.log` / `system.log`
- run meta: `run.json`
