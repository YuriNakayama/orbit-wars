# Reinforce Case6 — Prioritized Fictitious Self-Play (PFSP)

`pipeline/reinforce/case6/` は強化学習ファミリの **自己対戦 (self-play) / PFSP**
ケース。case3 で確立した学習レシピ (head=PerPlanetHead from_head なし、
shaping=planets/0.50、lr decay、episodes=128、gamma 0.995、entropy 0.02、
target_kl 0.02、200 iter) を **そのまま継承**し、**対戦相手の構成だけ**を変える。

現在のルールベース相手 (curriculum: noop → baseline_jax_lite) に加え、
既に JAX 化されている他エージェント (baseline_jax_full) や **自分自身の過去
snapshot** を相手に自己対戦させ、Prioritized Fictitious Self-Play (PFSP) で
より強いエージェントを作成できるかを検証する。

実験仮説リスト: [`docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md`](../../../../docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md)

## 対戦相手モード (rollout_jax.py)

| mode | 内容 | 由来 |
|------|------|------|
| `noop` | 何も発射しない | case3 から継承 |
| `baseline_jax_lite` | JAX ルールベース (baseline_v1 相当) | case3 から継承 |
| `baseline_jax_full` | JAX ルールベース (full) | case3 から継承 |
| `self_snapshot` | **学習開始時点の凍結 self snapshot を決定論で対戦 (H1 で追加)** | case6 新規 |

H1 では snapshot を学習開始 model に固定。pool 化 / 周期更新 / 優先度 sampling は H2 以降。

## アーキテクチャ概要 (= case1 と同一)

- **backbone**: Set Transformer (ISAB×4, hidden=192, attn_heads=8, inducing=24)
- **head_per_planet**: per-source × (P+1) pointer-attention categorical
  (`MAX_PLANETS=36`, no-op sentinel slot あり) + `ship_pred` 回帰スカラ。
  **from_head は持たない** (case2 との唯一の差分)。
- **value_head** (ctx → scalar V) と **ship_log_std** (state-independent Gaussian σ)。

action モデル:
- target: `Categorical(per_planet_logits)` over `(P+1)` slot per source planet
- ships: `Normal(ship_mean, exp(ship_log_std))` over `log1p(ships)` per source

PPO 損失 = clipped surrogate + value MSE − entropy bonus (本 case は BC OFF /
KL anchor なし、β=0)。

## 継承した「from_head 以外の有用な工夫」

| 工夫 | 値 | 出典 (実験) |
|------|-----|-----------|
| 報酬 shaping_mode | `planets` (領域支配 Δ) | F: ships → planets |
| shaping_coef | 0.50 | V': trend 最高 +0.145、1.0 は over-shaping |
| opponent | curriculum (noop → baseline_jax_lite) | B2 系 |
| switch_iter | 5 (baseline 期最大化) | H4: trend +0.204 で optimum |
| lr | 3e-5 → 3e-6 線形 decay | H + D: per-update Δ 縮小 |
| episodes_per_iter | 128 | S: sample variance 1/2 |
| gamma / entropy_coef / target_kl | 0.995 / 0.02 / 0.02 | A2 / G |
| iterations | 200 (長期学習) | case1 Z v2 が同手法で last-10 0.491 |

> 学習レシピの実値は `configs/kaggle_jax_train.yaml` を参照。

## ディレクトリ構成

```
pipeline/reinforce/case3/
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
│   ├── env.py               OrbitWarsEpisode + shaping (PyTorch 経路)
│   ├── rollout.py / ppo.py / train.py   PyTorch PPO (legacy、parity 参照用)
│   └── *_jax.py             JAX PPO (本番経路、下記「JAX 化」参照)
├── configs/                 *_jax.yaml が現役。kaggle_jax_train.yaml が canonical
└── evaluation/eval_vs_baseline.py
```

> 学習レシピの実値は本番 config `configs/kaggle_jax_train.yaml` を唯一の正とする
> (上記「継承した工夫」表と一致)。`smoke.yaml` / `train.yaml` 等の PyTorch config
> は legacy parity 用で、値が異なる。

## 手順 (JAX 学習、本番)

```bash
cd bot

# 1) CPU smoke で配線確認 (1 iter × 4 ep × 50 step)
uv run python -m pipeline.reinforce.case3.training.train_jax \
    --config pipeline/reinforce/case3/configs/train_jax_smoke.yaml

# 2) 本番学習は RunPod GPU で (kaggle_jax_train.yaml = 200 iter)
git push origin <branch>
dev/runpod train "$(git rev-parse HEAD)" --case case3 --watch

# 3) 評価 (vs baseline, 100 戦)
uv run python -m pipeline.reinforce.case3.evaluation.eval_vs_baseline \
    --episodes 100 --baseline baseline_v1
```

> JAX `best.pt` は npz 形式。PyTorch inference (`policy/agent.py`) への再ロードは
> 未対応 (follow-up)。PyTorch 経路で学習する場合のみ `training/train.py` を使う。

agent registry: `rl_v3` として `bot/src/dataset/selfplay/agents.py` に登録。

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
pipeline/reinforce/case3/
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
uv run python -m pipeline.reinforce.case3.training.train_jax \
  --config pipeline/reinforce/case3/configs/train_jax_smoke.yaml

# 本番 (5 iter × 16 ep × 500 step、BC warm-start + kl anchor)
uv run python -m pipeline.reinforce.case3.training.train_jax \
  --config pipeline/reinforce/case3/configs/train_jax.yaml

# RunPod GPU bench (現状の wall-clock を pod で計測)
git push origin <branch>
dev/runpod train <sha> --case bench_rollout_gpu --cloud-type ALL
```

train_jax の出力は PyTorch 版 (`training/train.py`) と並走可能。`best.pt`
は numpy npz として保存される (JAX leaves を直接 dump)。PyTorch inference
パスへの再ロードは未対応 (本 PR スコープ外、follow-up)。

