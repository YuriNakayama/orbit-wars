# Evaluation System — Requirements Definition

## 背景と目的

Orbit Wars 用の bot 開発において、**「ローカルで高速に対戦を回す → データを蓄積する → 分析する → 再生（可視化）する」** の4工程をひとつの評価基盤として提供する。既存の `pipeline/case1/evaluation/selfplay.py` は自己対戦のみ・逐次実行・蓄積が JSON 直書きという制約があり、バージョン比較や Kaggle 投稿前のレーティング推定には使えない。

本機能の主目的:

1. **エージェント強さ比較** — 新旧バージョン（例: `baseline_v1` vs `baseline_v0`）や異実装エージェント（`baseline` vs `case0`）の勝率を統計的に評価する。
2. **デバッグ用リプレイ可視化** — 敗戦やバグケースの観測をノートブック上で `env.render("ipython")` で再生し、行動を分析する。

副次的な目的:

- タイムアウトや 1 ターン時間分布を可視化して Kaggle 提出前のリスクを検出。
- 惑星所有推移・ミッション分布など、エージェントの癖を数値で把握。
- 将来的に RL / 模倣学習に使える質の高い軌跡データを副産物として蓄積（Phase2 以降）。

## ユーザーストーリー

- **As a bot 開発者、As me**、私は複数の agent 実装を CLI 1コマンドで対戦させ、勝率を即座に確認したい。Kaggle 提出前の判断を加速するため。
- **As a bot 開発者、As me**、私はリプレイを Jupyter ノートブック上で再生し、敗戦シーンの原因を特定したい。ルールベースの調整を正確に行うため。
- **As a bot 開発者、As me**、私は蓄積した対戦データを Polars/pandas で集計し、タイムアウト率・平均ターン数・ミッション分布などを一括レポート化したい。
- **As a bot 開発者、As me**、私は多数エピソード (N=数百〜千) を multiprocessing で並列実行し、数分で結果を得たい。

## 機能要件

### FR1. 汎用対戦ランナー
- `python -m env.cli run --agents NAME1,NAME2[,NAME3,NAME4] --mode {1v1,ffa4} --episodes N --seed S --parallel P [--save-replay/--no-save-replay]` を提供する。
- エージェント名は `src/env/agents.py` の `AGENT_REGISTRY: dict[str, Callable]` で解決。
- `--parallel P` 指定時は `multiprocessing.Pool(P)` で episodes を並列実行。`P=1` は逐次。
- 1v1 は 2 agents、ffa4 は 4 agents 必須。足りない場合は CLI エラー。
- 各エピソードは seed=`base_seed + i` で独立シード。

### FR2. エージェントレジストリ
- `src/env/agents.py` に `AGENT_REGISTRY` を定義。初期登録:
  - `baseline_v1` → `pipeline.case1.baseline.agent:agent`
  - `case0` → `pipeline.case0.main:agent`
  - `random` → `kaggle_environments` 標準の "random" エージェント（文字列参照可）
- 登録には git commit SHA を取得して `agent_version` として記録する機能を持つ（再現性のため）。

### FR3. マッチ記録の二層蓄積
- **layer-1 (集計)**: `data/matches/index.parquet` に 1 match 1 行で append。hive partition by `mode`（`mode=1v1/`, `mode=ffa4/`）。
- **layer-2 (リプレイ)**: `--save-replay` 指定時、`data/matches/replays/{match_id}.json.gz` に `env.toJSON()` を gzip 保存。省略時はサイズゼロ。
- `match_id` は `ULID` または `{timestamp}_{short_sha}_{seed}` 形式。
- `index.parquet` のスキーマ:
  | 列 | 型 | 説明 |
  |---|---|---|
  | match_id | str | マッチID |
  | run_id | str | CLI 呼び出し単位（同一実行の括り） |
  | mode | str | "1v1" / "ffa4" |
  | seed | int | エピソードシード |
  | started_at | datetime | 実行開始時刻 UTC |
  | elapsed_sec | float | 実行所要時間 |
  | turns | int | 実行ターン数 |
  | winner | int | 勝者 player id (-1=draw) |
  | draw | bool | 引き分け |
  | agent_0_name, agent_1_name, agent_2_name, agent_3_name | str | 各 player のエージェント名（不在は空文字） |
  | agent_0_version, ..., agent_3_version | str | git sha / タグ |
  | agent_0_score, ..., agent_3_score | int | 最終 ship 合計（所有惑星 + 飛行中） |
  | agent_0_timeouts, ..., agent_3_timeouts | int | actTimeout 超過ターン数 |
  | agent_0_turn_p50, _p95, _max | float | 1ターン時間の分位数（秒） |
  | replay_path | str | リプレイファイル相対パス（未保存は空） |
  | git_sha | str | リポジトリの現在 HEAD |

### FR4. 分析ユーティリティ
- `src/env/analyze.py` に以下の関数を提供:
  - `load_matches(filters: dict | None = None, root: Path = DATA_ROOT) -> pl.DataFrame` — `index.parquet` を hive 付きで scan。
  - `agent_winrate(df, opponent_filter=None) -> pl.DataFrame` — agent × 対戦相手の勝率マトリクス。
  - `timing_distribution(df) -> pl.DataFrame` — agent 別の p50/p95/max タイムアウト率。
  - `mission_distribution(df) -> pl.DataFrame` — 利用可能なミッションログが入った場合の集計。ミッションログは Phase2（FR7 参照）。
- `src/env/report.py` で CLI から `rich.Table` を使って勝率・タイムアウト・平均ターン数のサマリを標準出力する。

### FR5. リプレイ再生
- `src/env/replay.py` に `load_replay(match_id: str) -> Environment` を提供。内部で `json.loads + gzip.decompress + make("orbit_wars", configuration=..., steps=...)`。
- `pipeline/case1/eda/replay_viewer.py`（percent-format）をサンプルとして配置。1セルで `env = load_replay(mid); env.render(mode="ipython")` が動くこと。

### FR6. 観測可能性
- CLI は `rich.progress.Progress` の進捗バーで「n/N episodes 完了」を表示。
- 実行完了時、`rich.Table` で `agent_name × wins × win_rate × avg_turns × timeout_rate` を出力。
- `logging` 経由で構造化ログ（print禁止）。verbose オプションで DEBUG レベル。

### FR7. （Phase2）ミッション・タイムラインロガー
- baseline エージェントがどの mission (capture/snipe/swarm/reinforce/crash_exploit) を選んだかをターン別にログに出す機構。
- 本機能ではスキーマだけ予約しておき、実装は後続イテレーションで行う（`agent_X_mission_counts` 列を予約）。

### FR8. 既存 `pipeline/case1/evaluation/` の `src/env/` への移行
- `pipeline/case1/evaluation/selfplay.py` を削除し、新実装 `src/env/` に集約。
- `pipeline/case1/evaluation/snapshot_update.py` は **テスト用スナップショット更新の役割が別物**のため、`src/env/` には移さず `pipeline/case1/evaluation/` にそのまま残す（または `tests/pipeline/case1/` 配下に移動）。Step 4 で再判断。

## 非機能要件

- **観測可能性**: 進捗バー + rich テーブルで結果の即時確認（**必須**）。
- **再現性**: seed・agent_name・agent_version・git_sha を全マッチに記録（**推奨**）。seed 同一でも完全決定性は保証されない（Orbit Wars 環境の制約）が、「再実行したら近似再現される」レベルを目標とする。
- **スループット**: 8コア並列で 1v1 N=1000 エピソードを 5 分以内（**目標値、厳格 SLA ではない**）。
- **ストレージ効率**: `--no-save-replay` でリプレイ 0 バイト、保存時は gzip で平均 1〜5 KB/episode（**推奨**）。
- **コード品質**: ruff / mypy strict パス、backend.md / pipeline.md 規約に準拠。テストカバレッジ 80%+ 目標。

## 対象外（Out of Scope）

- **Web UI / Streamlit / Dash** — 要件になし。ノートブック + `env.render` に限定。
- **クラウドストレージ / MLflow** — ローカルファイルのみ。
- **自前高速シミュレータ** — kaggle_environments に依存する。
- **openskill レーティング** — Phase2 以降。MVP は勝率のみ。
- **Kaggle submission との統合** — `src/submit/` とは完全に独立。
- **FFA4 専用の特殊ロジック**（例: 3v1 の扱い、team 対戦など）— 平等にサポートするが特別扱いしない。

## 用語集

| 用語 | 定義 |
|---|---|
| match | 1 エピソード = 1 対戦（`env.run` の1呼び出し）。本文書での最小単位 |
| run | 1 回の CLI 呼び出し（N マッチ包含）。`run_id` で括られる |
| agent_name | `AGENT_REGISTRY` のキー文字列 |
| agent_version | 記録時点の `git rev-parse --short HEAD` |
| replay | `env.toJSON()` をそのまま gzip JSON にしたもの |
| index | `data/matches/index.parquet` のマッチ集計テーブル |
| mode | `"1v1"` または `"ffa4"` の文字列 |
