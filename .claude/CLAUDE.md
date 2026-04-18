# Kaggle Orbit Wars: Bot Agents

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) 参戦プロジェクト。`kaggle-environments` の Orbit Wars 環境上で、1v1 / 4人FFA に対応するAIエージェントを開発する。コンペの詳細仕様は [`docs/competition/abstract.md`](../docs/competition/abstract.md) を参照。

## Agent Pipeline

```
observation (planets, fleets, comets, player, ...)
  ↓
Feature Extraction    (軌道予測・脅威評価・生産ポテンシャル)
  ↓
Policy                (ルールベース / 学習済みモデル)
  ↓
Action Selection      [[from_planet_id, angle, num_ships], ...]
  ↓
kaggle_environments   env.step()
```

## Technology Stack

- **Language**: Python 3.13
- **Simulator**: `kaggle-environments` (Orbit Wars env)
- **Numerics**: NumPy, Pandas, Polars
- **AI / RL** (optional): PyTorch, Stable-Baselines3 などを必要に応じて追加
- **Testing**: Pytest + pytest-cov, Ruff, Mypy
- **Package Management**: UV

## Folder Structure

```
src/
  agents/               提出用エージェント（main.py がエントリポイント）
  env/                  kaggle-environments ラッパー・自己対戦ユーティリティ
  features/             観測→特徴量変換、軌道予測
  policies/             ルールベース / 学習済みポリシー
  utils/                共通ユーティリティ
pipeline/
  case1/                戦略別の学習・評価パイプライン
    eda/                観測データの探索的分析
tests/                  Pytest unit tests
data/                   リプレイ・学習ログ（大きいものは gitignore）
dev/                    Development scripts
docs/
  competition/          コンペ仕様まとめ（abstract.md 等）
  plans/                Feature plans
  research/             Research prompts and outputs
```

## Commands

```bash
dev/setup            # Install dependencies (uv sync)
dev/format           # Code formatting (ruff)
dev/lint             # Static analysis (ruff + mypy)
dev/test-backend     # CI (format check → lint → type check → pytest)
dev/create-worktree  # Create git worktree with .env copy
```

## Kaggle Submission Policy

Any real remote submission (`uv run python -m submit submit`, `dev/submit`, `kaggle competitions submit`, the `cd-kaggle-submit.yml` workflow_dispatch) is irreversible and consumes the daily 5-submission quota — always obtain explicit user approval immediately before executing, showing the case / message / mode to be submitted. Dry-run, archive build, and read-only history checks do NOT require approval. Prior approval covers only that single submission and does not extend to later submissions or auto-mode / autonomous loops.

## Glossary

| Term | Description |
|------|-------------|
| Orbit Wars | Kaggle主催のシミュレーション対戦コンペ。Planet Warsの現代版 |
| Planet | `[id, owner, x, y, radius, ships, production]` 形式。静止惑星 / 軌道惑星 |
| Fleet | `[id, owner, x, y, angle, from_planet_id, ships]`。速度は艦数に依存 |
| Comet | ターン 50/150/250/350/450 に4つ1組で出現する移動惑星 |
| Home Planet | プレイヤーの初期所有惑星（初期艦数10） |
| Skill Rating | 提出ごとの N(μ, σ²) レーティング。勝敗のみで更新 |
| Overage Time | 各エピソードで共有される追加思考時間バジェット |

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/backend.md` | `src/**`, `tests/**` | Python実装、pytest、ruff/mypy設定 |
| `.claude/rules/pipeline.md` | `pipeline/**` | 自己対戦・学習・評価パイプライン |
| `.claude/rules/infra.md` | `infra/**` | Terraform / クラウド学習基盤（使用時のみ） |
| `.claude/rules/security.md` | Always loaded | コミット、シークレット、CI/CD |

## Response Language

- Internal reasoning should be in English
- All user-facing output must be in Japanese（全てのユーザー向けの出力は日本語で行うこと）
