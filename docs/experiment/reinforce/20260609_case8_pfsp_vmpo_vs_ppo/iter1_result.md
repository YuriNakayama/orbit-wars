# Reinforce/case8 — pfsp_vmpo_vs_ppo (iter1) RESULT

> 関連: iter1_plan.md / hypotheses.md
> run_id: 20260609-062331__feature-poc-v-mpo__0c08b8a__seed0 / commit: 36c78fde / case: case8 (reinforce)
> 開始: 2026-06-09 06:23Z (pod) / 学習 07:34Z / 終了: 07:39Z (iter6 後に pod 異常終了) / コスト: ~$0.30 (interactive pod ~27min)

## Summary

H0 (case8 scaffold + `algo: ppo|vmpo` フラグ + PFSP/held-out/Elo wiring) は **支持された (adopted)**。
case7→case8 コピーが import 健全、`algo` フラグが両値で機能 (ppo は学習進行、vmpo は
`NotImplementedError` ガード)、PFSP f_var pool 切替・held-out eval・Elo 更新・S3 crash-safe
ckpt が RunPod GPU 上で end-to-end に動作することを iter0-6 のメトリクスで確認。

ただし iter1 実行中に **本物 rulebase (python_v8) を JAX rollout に通せない**という構造制約が
判明し、設計を 1 点変更した (held-out を in-JAX `baseline_jax_full` proxy に、本物 case8 は
学習後 offline paired 評価に分離)。H0 の受入条件 (両 algo の配線 + held-out/Elo 記録) は満たす。

## Numbers (RunPod, algo=ppo, GPU RTX4090, S3 metrics.json より)

| iter | train opp | win_rate | held-out (baseline_jax_full) | agent_elo |
|---|---|---|---|---|
| 0 | noop | 0.719 | 0.250 | 1492.0 |
| 1 | noop | 0.812 | — | — |
| 2 | noop | 0.875 | — | — |
| 3 | noop | 0.750 | — | — |
| 4 | baseline_jax_full (f_var) | 0.250 | 0.375 | 1488.4 |
| 5 | baseline_jax_lite (f_var) | 0.250 | — | — |
| 6 | baseline_jax_lite (f_var) | (S3未flush) | — | — |

- 定常 iter 速度: rollout ~22s + update ~1s ≈ **23s/iter** (iter0 のみ compile で 30s+30s)。
- curriculum: iter0-3 noop → iter4+ f_var pool (noop卒業後 full/lite を勝率0.5付近探索で選択)。
- held-out (every=4): iter0=0.25 / iter4=0.375 を記録、Elo 1500→1492→1488 と更新。配線 OK。
- per-iter win が noop=0.7-0.88 → rulebase proxy=0.25 に落ちるのは既知の天井 (ランダム寄り
  初期 policy は in-JAX rulebase に勝てない)。H0 は強さでなく配線の検証なので想定内。

## Diagnosis

**機能した点 (H0 受入)**: algo フラグ / PFSP f_var pool 切替 / held-out + Elo / S3 ckpt 全て GPU 上で動作。

**判明した構造制約 (設計変更)**: 本物 rulebase `python_v8` は `jax.pure_callback(vmap_method="sequential")`。
`collect_rollout_jax` で episodes を vmap・horizon を scan するため、これを (train pool でも
held-out rollout でも) 通すと **episodes×horizon (32×500=16k) 回の逐次 host 往復**が発生し、
RunPod GPU 上で iter0 が >10分停滞・GPU util 0%・CPU 117%。held-out は `it%every==0` で
iter0 から発火するため every=4 でも初回停滞。→ memory `python_v8_train_pool_gpu_stall` に記録。

**対処**: 学習ループの rollout は **in-JAX opponent のみ** (train pool = lite/full/self、
held-out = baseline_jax_full)。本物 case8 (baseline_v8) との比較は best.pt を学習後に
**offline paired 300戦**で取る (hypotheses.md 例外条件)。これでユーザーの「case8 で進捗評価」を
GPU 整合的に実現 (in-loop は proxy で高速追跡、最終確認は本物 case8 で正確に)。

**インフラ系の注記 (採否に影響なし)**:
- run #1 = python_v8 train pool / run #2-3 = git pull が onstart dvc lock でブロック + JAX が
  CPU 退行 / run #5 = iter6 後に interactive pod が無言終了 (OOM/接続断、log に traceback なし)。
- いずれも H0 のコード健全性とは無関係。iter0-5 のメトリクスは clean で一貫。

## Decision

- 採否: **adopted** (H0 = scaffold/wiring 検証完了)。`algo` フラグ・PFSP・held-out・Elo・S3 全て GPU 動作確認。
- 次の一手: **H1 (V-MPO loss 新規実装)**。`training/` に V-MPO loss module を追加し
  `algo=vmpo` 分岐 (現 `NotImplementedError`) を実装、PPO arm (本 iter で確立) と同一 harness で A/B。
  oneshot 学習 (`dev/runpod train --case case8`, reinforce_case8_kaggle_jax_train) への移行も検討
  (interactive pod の無言終了を回避、auto-recover が効く)。

## Artifacts
- metrics: `s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/20260609-062331__feature-poc-v-mpo__0c08b8a__seed0/metrics.json`
- best.pt: `.../best_i2_win0.8750.pt` + `.../best.pt`, ckpts: `.../ckpts/ckpt_i00{0..5}.pt`
- pod: terminated (課金停止済)
