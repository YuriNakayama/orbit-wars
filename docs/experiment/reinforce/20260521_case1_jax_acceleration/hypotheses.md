# Hypotheses — reinforce/case1 JAX acceleration

> 作成日: 2026-05-18
> 最終更新: 2026-05-21
> 状態: completed
> 最大 iteration: 1 (大型ジャンル — featurizer/rollout/PPO の段階的 JAX 化)
> 主要メトリクス: iter1 wall-clock (16 ep × 500 step) と PyTorch baseline (270s) 比の speedup
> 既定 episode 数: 学習 5 iter smoke

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule)
- ローカル self-play 300 対戦は実施しない (この iter は速度実験で勝率を見ない)
- BC win-rate ±2pp 検証も skip (W6-a smoke で BC parity 1e-6 達成済、機能等価)

### 分析
- replay 分析は実施しない (速度実験のため戦術 replay 分析は無関係)

### 実行
- 既存 PyTorch rollout は dev/runpod train で動作するので、回帰確認のみ
- RunPod の cost cap: $1.00 (bench 系の retry に複数 GPU 投入する可能性があるため通常 $0.5 より緩める)

### 例外条件
- W6 残り (b/c) は GPU 実測値次第で判断。10s 未満なら不要

## 仮説リスト (priority 順)

- [x] (P1) **H1: featurizer JAX 化で 5-10× 高速化**
  — 545 LoC の Python loop を jnp ベクトル化 + vmap で大幅高速化期待。
  **adopted** (W1-W2f 完了、vmap(16) 5.7× 達成、parity 1e-4)
- [x] (P1) **H2: ActorCritic Equinox 移植で host call 削減**
  — PyTorch forward は host call。Equinox にすれば全体を device に。
  **adopted** (W3 完了、param count 3.1M で一致、forward parity 1e-6)
- [x] (P1) **H3: rollout を vmap(16) で並列化**
  — lax.scan over horizon、vmap over episodes で iter1 ≤ 30s 目標。
  **adopted** (W4 完了、M-series CPU で 16.7s = 16× 達成)
- [x] (P2) **H4: PPO update も Optax + Equinox autograd で device 内**
  — backward 経路を JAX に閉じ込めて end-to-end JAX 化。
  **adopted** (W4-c + W6-a 完了、PPO update 18s → 3.1s = 5.8×)
- [ ] (P2) **H5: env.reset を JAX 化して vmap 内 reset**
  — 現状 reset は host (Python random)。jit 内 reset で更なる host call 削減。
  **deferred** (reset は iter コストの数%、ROI 低い)
- [ ] (P3) **H6: 学習ループ全体を lax.scan に閉じ込め**
  — iter ループ自体を device に。最終形。
  **deferred** (iter ループは 5 回程度、measurable overhead 小)

## Iteration log

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-18 | H1-H4 | (no separate plan) | local M-series + A100 80GB | iter1 = 16.7s / 27.6s | adopted | iter1_result.md | (analysis 未実施) |

## 参考 (References)

- `docs/plans/jax-env/01-design.md` — JAX env + Phase C plan
- `bot/src/jax_env/` — Phase B implementation (590 + 23 tests)
- `bot/pipeline/reinforce/case1/policy/featurizer_jax.py` — W1-W2 implementation
- `bot/pipeline/reinforce/case1/policy/model_jax.py` — W3 Equinox ActorCritic
- `bot/pipeline/reinforce/case1/training/{rollout,ppo,train}_jax.py` — W4-W5
- bench results: `data/output/models/reinforce/rollout_gpu/runs/20260520-133515_*`
