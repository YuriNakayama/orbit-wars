# Orbit Wars: Kaggle Bot Competition

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) 参戦リポジトリ。1v1 / 4人FFA のリアルタイム戦略シミュレーション上で対戦する **AIエージェント** を開発します。

## 大会概要

- **主催**: Kaggle (Bovard Doerschuk-Tiberi, Walter Reade, Addison Howard)
- **種別**: Featured Simulation Competition
- **賞金総額**: $50,000（1〜10位にそれぞれ $5,000）
- **目的**: 2010年の Planet Wars を現代化した対戦ゲームで、他参加者のボットと対戦し勝率を最大化する

### エージェントに求められる能力

| 能力 | 説明 |
|------|------|
| **盤面認識** | 惑星・フリート・コメット・軌道運動を解釈 |
| **行動選択** | 1ターン1秒以内に `[from_planet_id, angle, num_ships]` のリストを返す |
| **軌道予測** | 軌道惑星（公転）とコメット（直線移動）の将来位置を予測 |
| **戦闘判定** | グループ化された戦闘ルールを踏まえて勝敗と占領を推定 |

## ゲームルール概要

- **ボード**: 100×100 の連続2D空間、中心 (50,50) に半径10の太陽（通過したフリートは消滅）
- **惑星**: 20〜40基。4折対称配置で、軌道惑星は 0.025〜0.05 rad/turn で公転
- **コメット**: ターン 50/150/250/350/450 に4つ1組で出現、速度 4.0
- **勝利条件**: 500ターン終了時、または相手全滅時に `自軍惑星の艦数 + 飛行中フリートの艦数` が最大のプレイヤーが勝利
- **戦闘**: 攻撃者をowner別にグループ化し、最大勢力vs第2勢力の差分が駐留艦と対戦

詳細は [`docs/competition/abstract.md`](docs/competition/abstract.md) を参照。

## 大会スケジュール

| 日程 (UTC 23:59) | イベント |
|------------------|---------|
| 2026-04-16 | 開始 |
| 2026-06-16 | エントリー / チームマージ締切 |
| 2026-06-23 | 最終提出締切 |
| 2026-06-24 〜 07-08 頃 | 追加対戦期間（収束まで継続） |

### 評価方式

各提出は **ガウス分布 N(μ, σ²)** のスキルレーティングを持ち、対戦結果で更新される。

- 初期値 μ₀ = 600、対戦を重ねると σ が縮小
- 更新は勝敗のみを参照（スコア差は無関係）
- 1日最大5提出、最新2件が最終提出候補

## Technology Stack

- **Language**: Python 3.13
- **Simulator**: `kaggle-environments` (Orbit Wars env)
- **Numerics**: NumPy, Pandas, Polars
- **AI / RL**: PyTorch / Stable-Baselines3 など（必要に応じて導入）
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
  rulebase/             ルールベース戦略パイプライン
    case0/              単純スナイパー (参考実装)
    case1/              baseline_v1
      eda/              観測データの探索的分析
    case2/              baseline_v2
  imitation/            模倣学習パイプライン
    case1/              DeepSets BC
tests/                  Pytest unit tests
data/                   リプレイ・学習ログ（大きいものは gitignore）
dev/                    Development scripts
  setup                 Install dependencies (uv sync)
  format                Code formatting (ruff)
  lint                  Static analysis (ruff + mypy)
  test-backend          Backend CI (format check -> lint -> type check -> pytest)
  create-worktree       Create git worktree with .env copy
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
dev/test-backend     # CI (format check -> lint -> type check -> pytest)
dev/create-worktree  # Create git worktree with .env copy
```

## Evaluation Framework (`src/env/`)

ローカルでの対戦実行・データ蓄積・分析・再生を提供する汎用フレームワーク。

```bash
# 対戦実行 (結果は data/matches/ に保存)
uv run python -m env run \
  --agents baseline_v1,case0 --mode 1v1 -n 10 --parallel 4

# 最新 10 件を一覧
uv run python -m env list --mode 1v1 --limit 10

# 指定 match_id のリプレイを検査
uv run python -m env replay-inspect <match_id>
```

- **データ**: Parquet (hive partition: `mode=`) に指標、`replays/{match_id}.json.gz` に env.toJSON。
- **分析**: `env.analyze.agent_winrate(...)` / `timing_distribution(...)` / `mode_summary(...)` を呼ぶ。
- **可視化**: `pipeline/rulebase/case1/eda/replay_viewer.py` を Jupyter / VS Code で開き、`env.render("ipython")` を実行。

## Glossary

| Term | Description |
|------|-------------|
| Orbit Wars | Kaggle主催のシミュレーション対戦コンペ（Planet Wars の現代版） |
| Planet | 盤面上の惑星。軌道惑星（公転）と静止惑星がある。生産量 1〜5 |
| Fleet | `[id, owner, x, y, angle, from_planet_id, ships]` 形式の飛行中艦隊 |
| Comet | ターン 50/150/.../450 に出現する移動惑星。占領・生産可能 |
| Home Planet | 各プレイヤーの初期所有惑星（初期艦数10） |
| Skill Rating | 提出ごとのガウス分布 N(μ, σ²) によるレーティング |

## Links

- [Kaggle: Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
- [コンペ概要まとめ](docs/competition/abstract.md)
