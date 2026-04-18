# Evaluation System — Codebase Research

本機能は「ローカル上で高速に対戦実行」「対戦データ蓄積」「対戦データ分析」「対戦データ再生（可視化）」の4要素を case 横断で提供する評価基盤である。以下、現行コードベースの該当領域を深堀りした結果をまとめる。

## 1. Deep Codebase Analysis

### 1.1 `pipeline/case1/evaluation/selfplay.py` — 現行の自己対戦 CLI

- **Files analyzed**: `pipeline/case1/evaluation/selfplay.py`（全234行）
- **Current implementation**:
  - `typer` ベース CLI（`app = typer.Typer`）、`run` コマンド1つ。
  - `_make_env(mode, seed)`: `kaggle_environments.make("orbit_wars", configuration={"agents": N, "seed": seed})` を返す。`mode` は `"1v1"` / `"ffa4"`。
  - `_agent_with_timing(timings)`: baseline agent を `time.perf_counter()` でラップし、1ターン毎の経過秒をリストに append。タイムアウト判定は `> 1.0s`。
  - `_episode_winner(final_state)`: `ACTIVE` プレイヤーが1人なら勝者、それ以外は `reward` 最大を勝者。
  - `_player_scores(env)`: 最終ターンの observation から `planets.ships(所有者一致) + fleets.ships(所有者一致)` を合計。
  - 出力: `data/replays/case1/{YYYYMMDD_HHMMSS}/episode_{i}.json`（`env.toJSON()` があれば利用、なければ `env.steps`）、`episode_{i}_timings.json`、`summary.csv`。
- **Key interfaces**:
  - 入力: `episodes: int, mode: str, seed: int, output_dir: Path, save_replay: bool, verbose: bool`
  - 出力: stdout に `rich.Table`、ディレクトリに JSON / CSV。
- **Patterns used**: `typer` + `rich.console` + `rich.logging.RichHandler` + `rich.table.Table`。ルール `pipeline.md` / `backend.md` に適合。
- **Coupling & side effects**:
  - `baseline_agent` を直 import しており、**他 agent との対戦ができない**（自己対戦のみ）。
  - 1プロセス逐次実行。`multiprocessing` 未使用。
  - 出力ディレクトリはタイムスタンプ固定なので、同時起動すると別ディレクトリになる（概ね安全）。
  - `summary.csv` のフォーマットは4プレイヤー固定列（1v1 でも `p2_score` `p3_score` 空）。
- **Test coverage**: なし（`tests/pipeline/case1/` には agent のみ）。
- **Gaps identified**:
  - 対戦相手プール機構（旧バージョン vs 現行）未実装。
  - 並列化なし → N=100 を超えると遅い。
  - 蓄積データが JSON 直書きで、集計時に1ファイルずつ loads が必要（分析に不向き）。
  - 再生（可視化）する仕組み・ノートブックは未整備。
  - `summary.csv` にエージェント識別子（revision / git sha / bot_name）が無く、多対戦比較に使えない。

### 1.2 `pipeline/case1/evaluation/snapshot_update.py` — スナップショット更新

- **Files analyzed**: 全61行。
- **Role**: テスト用の (obs, action) スナップショットを `tests/pipeline/case1/snapshots/` に書き出す。評価基盤とは別目的。
- **Relevant fact**: `env.step([[], []])` で `turn+1` 回だけ進めて `env.steps[-1][0]["observation"]` を dict 取得 → 観測の形状が dict（`player, step, planets, fleets, angular_velocity, initial_planets, comets, comet_planet_ids`）であることが確認できる。

### 1.3 `pipeline/case0/main.py` — 参照用の単純 agent

- **Role**: 最寄り惑星 snipe エージェント。`agent(obs)` 単体関数。evaluation-system の「対戦相手プール」の最小候補として使える。
- **Note**: `obs` が dict / object の両方に対応する防御的 getattr/dict.get パターンが既出。

### 1.4 `pipeline/case1/baseline/` — ベースラインエージェント

- **Role**: 本命エージェント。`agent = pipeline.case1.baseline.agent.agent` がエントリポイント。`plan_moves(world)` が plan 生成の中核。
- **Relevance to evaluation**: 評価対象の一方としてそのまま呼べる。依存が `pipeline.case1.baseline.core.*` に閉じている。

### 1.5 `src/submit/` — Kaggle 提出パイプライン

- **Files**: `packager.py, validator.py, history.py, kaggle_api.py, auth.py, __main__.py`
- **Relevance to evaluation**: なし。Kaggle submission の archive/submit/history 管理のみ。評価システムとは独立する。

### 1.6 `pipeline/case1/notebook/lb-897-orbit-wars-2026-reinforce.py`

- Kaggle Notebook 書き出し。ruff 除外。評価システム設計の考慮対象外。

### 1.7 `data/` — 既存レイアウト

- `data/lake/` 空ディレクトリのみ（.gitkeep すら未配置）。`data/replays/case1/` は実行時に生成される想定で、現時点のリポジトリには無い。
- `.gitignore` には `log/` / `.env` 系は列挙されているが、`data/` 配下は明示除外されていない → リプレイが誤って commit されないよう、**`data/replays/**` と `data/lake/**` を gitignore に追加するのが望ましい**（pipeline.md の「大きいものは gitignore」方針に整合）。

### 1.8 `tests/` — 既存テスト

- `tests/pipeline/case1/test_baseline_agent.py`: 統合（`env.run([agent, agent])` が DONE で終わること）、スナップショット（obs → action 決定性）、合成 obs によるユニットの3種。
- `tests/submit/`: 提出パイプラインのテスト。
- **参考にすべきパターン**: `pytest.importorskip("kaggle_environments")` で kaggle_environments 未インストール環境をスキップ。`@pytest.mark.integration` マーカー使用（pyproject に registered）。

### 1.9 Dev scripts / CI

- `dev/test-backend`: 現行スクリプトは `cd "$(dirname "$0")/../backend"` と書かれており、リポ直下に `backend/` が存在しない（pyproject/src/tests はルートにある）。**既存バグあり**。評価システム作業の中で直す必要があるかはオーナー判断。`.github/workflows/ci-backend.yml` は未確認。
- `dev/lint`, `dev/format`: 他の backend scripts と合わせて ruff / mypy を呼ぶ想定（中身は未確認だが慣例どおり）。

## 2. Technical Constraints

| 区分 | 制約 | 根拠 |
|---|---|---|
| タイムアウト | 1ターン 1秒（`actTimeout=1`）。超過はログ化 | backend.md / pipeline.md |
| エピソード | 最大500ターン | abstract / docs |
| ログ | `print` 禁止、`logging` / `rich` | backend.md |
| 依存 | kaggle-environments には `orbit_wars` が含まれるバージョンを使う必要あり | docs/competition/20260418_evaluation.md |
| Python | 3.13、mypy strict、ruff line-length=88、max-complexity=5 | pyproject.toml |
| 並列化 | CPU bound 前提。`multiprocessing` 想定 | pipeline.md |
| シード | config に明記、再現可能に | pipeline.md |
| 出力 | `data/replays/{case}/{timestamp}.json`、実行毎にサブディレクトリ | pipeline.md |
| 免責事項 | Orbit Wars 環境は「同一シードでもエピソード長が変わる」という部分非決定性あり | snapshot_update.py docstring |
| Kaggle 提出 | 本機能では触らない（独立系統） | — |

## 3. Key Findings Summary

1. **母体となる資産は `pipeline/case1/evaluation/selfplay.py`** だが、自己対戦専用・逐次実行・JSON書きっぱなしで「対戦データ蓄積基盤」としては不足。
2. **ユーザーの方針**: 既存 selfplay.py は case1 固有として温存し、**`src/env/` 配下に case 横断の汎用評価フレームワーク（runner/recorder/loader/visualizer）を新規作成**する。
3. **ストレージ**: ローカルファイルのみ。JSON + Parquet（後述 web-research で検討）。
4. **可視化**: ノートブック上で `env.render()` を呼ぶだけに留める（独自UI不要）。→ リプレイ JSON は kaggle_environments が復元可能な形で保存すれば OK。
5. **並列化**: `multiprocessing` で CPU 並列エピソード。`asyncio` / 自前シミュレータは不採用。
6. **観測の形**: `obs` は dict (`player, step, planets, fleets, angular_velocity, initial_planets, comets, comet_planet_ids`)。この形で記録すれば再現・分析可能。
7. **テスト**: `pytest.importorskip("kaggle_environments")` と `@pytest.mark.integration` を踏襲する。
8. **ディレクトリ方針**: 新規フレームワークは `src/env/`（`env/` という既存命名に整合）。`pipeline/caseN/` からは `src.env.runner.run_match(...)` 的に呼ぶ。`data/matches/` を追加し、各対戦を `match_{id}.json[.parquet]` で保存、`index.parquet` でメタ集計する方針が候補。
