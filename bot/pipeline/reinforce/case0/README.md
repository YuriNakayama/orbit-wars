# Reinforce Case0 — PPO Baseline

`pipeline/reinforce/case0/` は強化学習 (RL) ファミリの最初のケースで、
PPO (Proximal Policy Optimization) による Orbit Wars エージェントの
ベースラインを提供する。

短期ゴールは「rule-based baseline 群と同じ評価フレームに乗せられる
学習済み RL エージェントを 1 種類用意する」こと。プレイ強度の追求は
後続 case (`reinforce/case1+`) に委ねる。

## ディレクトリ構成

```
pipeline/reinforce/case0/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ (greedy decode)
│   ├── model.py             # ActorCritic (DeepSets + 3 heads + value head)
│   ├── featurizer.py        # obs → BatchFeatures
│   ├── sampling.py          # 確率サンプリング / log_prob / entropy
│   ├── decoder.py           # SampledAction → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   └── weights.pt           # 学習済み重み (DVC 管理、git untracked)
├── training/                # 開発用 (.submitignore)
│   ├── env.py               # OrbitWarsEpisode (kaggle_env wrapper)
│   ├── rollout.py           # on-policy ロールアウト + GAE
│   ├── ppo.py               # PPO clipped surrogate update
│   └── train.py             # CLI entrypoint
├── configs/                 # smoke YAML (.submitignore)
│   └── smoke.yaml
└── evaluation/              # 開発用 (.submitignore)
    └── eval_vs_baseline.py
```

## 学習・評価

```bash
cd bot

# 1) 学習 (smoke config: 4 iter × 4 episode、CPU 数分)
uv run python -m pipeline.reinforce.case0.training.train

# 2) 重みを canonical 位置にコピー
cp data/output/models/reinforce/case0/runs/local_*/best.pt \
   pipeline/reinforce/case0/policy/weights.pt

# 3) 評価 (vs baseline_v1 を 100 戦)
uv run python -m pipeline.reinforce.case0.evaluation.eval_vs_baseline \
   --episodes 100 --baseline baseline_v1
```

## 設計原則

- **case 独立**: `pipeline/reinforce/case0/` は他 case を import しない。
  `geometry.py` も imitation/case1 から独立コピー。
- **Action 表現**: 3 ヘッド構造を踏襲。
  1. `from_head`: 自惑星ごとに Bernoulli (発射するか)
  2. `target_head`: per-source × planet の pairwise scoring
  3. `ships_head`: 4 buckets (25 / 50 / 75 / 100%)
  各 head は独立に sample / log_prob を計算し、合計する。
- **value head**: グローバルコンテキストから scalar V を回帰。GAE で
  advantage を計算 (γ=0.99, λ=0.95)。
- **報酬**: terminal +1/-1/0 + 軽い shaping `0.001 × Δ(my_ships - enemy_ships)`。
- **推論**: greedy argmax (`from_threshold=0.5`) + masking + 簡易 over-fire 抑制。

## 既知の制約

- 提出版 `agent` は重みファイル不在でもランダム初期化で動く (smoke 用)。
  実プレイ強度は学習を回した後の `weights.pt` に依存。
- Self-play は未実装。初期 baseline は `random_noop` (固定相手) に対する
  改善を確認するための smoke ループ。Self-play / opponent pool は case1 で扱う。
- Smoke config (4 iter × 4 ep) は学習収束目的ではなく E2E pipeline 動作確認用。

## モデルバージョン

| ファイル | 説明 |
|---------|------|
| `policy/weights.pt` | canonical。学習スクリプトの best.pt を手動 promote |

`rl_v0` は `bot/src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に登録済み。
