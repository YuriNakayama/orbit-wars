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
