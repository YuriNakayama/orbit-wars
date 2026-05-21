# Reinforce Case1 — PPO + BC Warm-start (case9 per_planet)

`pipeline/reinforce/case1/` は強化学習ファミリの 2 番目のケース。
**imitation/case9 の per_planet variant 重み** をベースに、PPO で
ファインチューンする RL ベースライン。

## アーキテクチャ概要

- **backbone**: Set Transformer (ISAB×4, hidden=192, attn_heads=8, inducing=24)
  — case9 BC checkpoint と完全一致。strict=False で 1:1 ロード可。
- **head_per_planet**: per-source × (P+1) pointer-attention categorical
  (`MAX_PLANETS=36`, no-op sentinel slot あり) + `ship_pred` 回帰スカラ
  — case9 同型。BC 重み 100% 流用。
- **新規追加**: `value_head` (ctx → scalar V) と `ship_log_std`
  (state-independent Gaussian σ パラメータ) のみ。PPO ウォームスタート時の
  ランダム初期化はこの 2 ヘッドのみ。

action モデル:
- target: `Categorical(per_planet_logits)` over `(P+1)` slot per source planet
- ships: `Normal(ship_mean, exp(ship_log_std))` over `log1p(ships)` per source

PPO 損失 = clipped surrogate + value MSE − entropy bonus + β·KL(π‖π_BC)

## ディレクトリ構成

```
pipeline/reinforce/case1/
├── main.py                  Kaggle entry
├── policy/                  提出物
│   ├── agent.py             greedy decode
│   ├── model.py             ActorCritic (backbone + per_planet + value + log_std)
│   ├── sampling.py          Categorical + Gaussian sample/log_prob/entropy
│   ├── decoder.py           SampledAction → action list (safety filter 内蔵)
│   ├── featurizer.py        obs → BatchFeatures (case9 と同型)
│   ├── candidates.py / templates.py / timeline.py / safety.py / geometry.py
│   ├── heads/               backbone.py / per_planet.py / value.py
│   ├── types.py             BatchFeatures / WorldSnapshot
│   └── weights.pt           学習済み重み (DVC 管理)
├── training/                開発用 (.submitignore)
│   ├── env.py               OrbitWarsEpisode + shaping
│   ├── rollout.py           GAE + on-policy buffer
│   ├── ppo.py               clipped surrogate + KL(BC) anchor
│   └── train.py             typer CLI
├── configs/
│   ├── smoke.yaml           4 iter × 4 ep smoke (opponent=random_noop)
│   └── train.yaml           50 iter × 8 ep 本番 (opponent=baseline_v1)
└── evaluation/eval_vs_baseline.py
```

## 学習レシピ

| 項目 | 値 |
|------|-----|
| BC 重み | `data/output/models/imitation/case9_per_planet/runs/20260512-080505__feature-imitation-data-volume-sweep__593a7c4__seed0/best.pt` |
| lr | 1e-4 (BC repr 保護のため case0 の 3e-4 より低め) |
| γ / λ | 0.99 / 0.95 |
| clip_eps | 0.2 |
| entropy_coef | 0.003 (BC prior が良いので低め) |
| **kl_beta** | **0.1** (frozen BC reference との KL anchor) |
| ppo_epochs | 4 |
| shaping_coef | 0.001 × Δ(my_ships − enemy_ships) |
| 報酬 | terminal ±1 + shaping |
| opponent | smoke は `random_noop`、本番は `baseline_v1` → self-play へ curriculum 予定 |

## 手順

```bash
cd bot

# 1) BC 重み取得 (初回のみ)
cd .. && uv run --project bot dvc pull data/output/models/imitation/case9_per_planet/runs/20260512-080505__feature-imitation-data-volume-sweep__593a7c4__seed0.dvc
cd bot

# 2) smoke training
uv run python -m pipeline.reinforce.case1.training.train

# 3) 重みを canonical 位置にコピー
cp data/output/models/reinforce/case1/runs/local_*/best.pt \
   pipeline/reinforce/case1/policy/weights.pt

# 4) 評価 (vs baseline_v1, 100 戦)
uv run python -m pipeline.reinforce.case1.evaluation.eval_vs_baseline \
    --episodes 100 --baseline baseline_v1
```

agent registry: `rl_v1` として `bot/src/dataset/selfplay/agents.py` に登録。

## 設計原則

- **case 独立**: case9 / case10 を import しない。policy 配下の依存モジュール
  (featurizer / candidates / templates / safety / geometry / heads/backbone /
  heads/per_planet) は **直接コピー** し、必要な改変のみ加える。
- **BC 重み互換**: `model.py` の state_dict キーは
  `backbone.*` / `head_per_planet.*` を case9 と完全に一致させる。
  追加した `value_head.*` / `ship_log_std.*` のみ random init。
- **fire 判定**: case9 と同様、no-op sentinel slot (index = MAX_PLANETS) に
  内包。Bernoulli from-head は持たない。
- **ships 表現**: Gaussian (log1p 空間)。BC の `ship_pred` を Gaussian 平均と
  して再利用。state-independent な scalar `log_std` のみ追加学習。
- **inference**: greedy argmax (target) + Gaussian 平均 (ships) で
  `agent.py` の `greedy_action` 経路を通る。
```

## JAX 化 (2026-05-21、PR #74)

PPO 学習を end-to-end JAX 化し、PyTorch baseline (270s/iter) に対して
**GPU で 17-18× の高速化** を達成。100-iter 学習が **7.5h → 25 min** に短縮。

### 速度比較 (16 ep × 500 step / iter)

| 環境 | wall-clock | speedup | 内訳 |
|---|---|---|---|
| PyTorch baseline (pod CPU) | 270s | 1.0× | rollout 247s + update 23s |
| JAX on M-series CPU (laptop) | 16.7s | **16×** | warm |
| JAX on RTX 4090 + W6-a (warm) | **15s** | **17-18×** | rollout 14s + update **0.54s** |

PPO update 単体は **92× 高速化** (50s → 0.54s)。warm iter は rollout 14s が
支配的で、scan body の per-step compute density (1.8 ms/step on RTX 4090)
が現状の上限。

### 実装の地図

PyTorch 版と並走する形で `*_jax.py` を新設:

```
pipeline/reinforce/case1/
├── policy/
│   ├── featurizer_jax.py      JAX featurizer (planet 41 + global 20 + template_ctx + candidate + timeline)
│   ├── model_jax.py           ActorCriticJax (Equinox 移植 + load_bc_weights_jax)
│   ├── sampling_jax.py        sample_action_jax + sampled_action_to_env_actions
│   └── sampling_eval_jax.py   evaluate_actions_jax (PPO update での log_prob 再計算)
└── training/
    ├── rollout_jax.py         collect_rollout_jax (vmap over episodes, lax.scan over horizon)
    ├── ppo_jax.py             ppo_update_jax (Optax + Equinox autograd, 単一 jit)
    └── train_jax.py           end-to-end PPO loop
```

依存: `equinox>=0.11`, `optax>=0.2`。GPU 実行時は `[dependency-groups] cuda`
で `jax-cuda12-plugin / pjrt + nvidia-cudnn-cu12>=9.8` を opt-in install
(`uv sync --no-dev --group env --group cuda`, Linux marker)。

### Parity 保証

- **JAX env vs Rust simulator**: trajectory parity 1e-5 (Rust と 500-step
  完全一致、622 件 jax_env tests)
- **JAX featurizer vs PyTorch**: tol=1e-4 で全 41 planet + 20 global + 40
  template_ctx + 14×8 candidate + 6 timeline 列が bit 一致 (74 件
  `test_featurizer_jax_parity.py`)
- **JAX model vs PyTorch ActorCritic**: forward output parity 1e-6
  (7 件 `test_model_jax_parity.py`、3.1M params 完全一致)

BC 重み (case9_per_planet) は `load_bc_weights_jax(model, weights_path)`
で PyTorch `.pt` から numpy 経由でロード可能。

### Phase の流れ

W1 → W2a-f → W3 → W4 → W4-c → W5 → W6-a の 7 段階。詳細は
`docs/experiment/reinforce/20260521_case1_jax_acceleration/iter1_result.md`
に記録 (Numbers / Diagnosis / Decision / Artifacts)。

W6-b (env.reset JAX 化) と W6-c (学習ループ lax.scan) は **deferred** —
rollout の真の bottleneck は scan body の per-step compute density で、
reset 高速化は ROI 低いと判明。さらなる速度向上を狙うなら bf16 化や
vmap batch 拡大が次の選択肢。

### 使い方 (JAX 学習)

```bash
cd bot

# CPU smoke (1 iter × 4 ep × 50 step)
uv run python -m pipeline.reinforce.case1.training.train_jax \
  --config pipeline/reinforce/case1/configs/train_jax_smoke.yaml

# 本番 (5 iter × 16 ep × 500 step、BC warm-start + kl anchor)
uv run python -m pipeline.reinforce.case1.training.train_jax \
  --config pipeline/reinforce/case1/configs/train_jax.yaml

# RunPod GPU bench (現状の wall-clock を pod で計測)
git push origin <branch>
dev/runpod train <sha> --case bench_rollout_gpu --cloud-type ALL
```

train_jax の出力は PyTorch 版 (`training/train.py`) と並走可能。`best.pt`
は numpy npz として保存される (JAX leaves を直接 dump)。PyTorch inference
パスへの再ロードは未対応 (本 PR スコープ外、follow-up)。

