# Log Data Collection — Codebase Research

Kaggle Orbit Wars 上位エージェントの対戦リプレイを収集するに当たり、既存の自己対戦評価基盤 (`src/env/`) と Kaggle CLI ラッパー (`src/submit/`) の資産を詳細に把握した結果をまとめる。新モジュールは既存構成と整合する形で `src/env/kaggle/` に追加する方針。

## Deep Codebase Analysis

### Area 1: 対戦記録パイプライン (`src/env/`)

- **Files analyzed**:
  - `src/env/runner.py:1-138` — `RunSpec` / `run_episodes`。mp.Pool による並列実行。
  - `src/env/executor.py:1-146` — `run_one_match(MatchSpec) -> (record_dict, replay_bytes|None)`。`env.toJSON()` → gzip 圧縮。
  - `src/env/recorder.py:1-73` — Parquet index + gzip リプレイ書き出し。hive 形式 `index.parquet/mode={mode}/run_{run_id}.parquet`。
  - `src/env/loader.py:1-58` — `list_matches()`, `load_replay_payload()`, `load_replay()` (kaggle_environments 再構成)。
  - `src/env/types.py:1-131` — frozen dataclass 群、`MAX_PLAYERS=4`, `SCHEMA_VERSION=1`, `MatchRecord.to_row()/from_row()`。
  - `src/env/cli.py:1-97` — typer CLI。`run` / `list` / `replay-inspect` サブコマンド。
  - `src/env/agents.py`, `analyze.py`, `report.py` — 補助モジュール。

- **Current implementation**: エージェント名リスト + mode + episodes + seed から `MatchSpec` を量産し、`multiprocessing.Pool` で `run_one_match` を並列実行する。ワーカー内で `kaggle_environments.make("orbit_wars", ...).run([...])` を呼び、`env.toJSON()` を gzip 圧縮して main に返す。main で `recorder.write_run(records, replay_bytes, data_root)` が Parquet と gzip を永続化。

- **Key interfaces**:
  - `MatchRecord` は flat な `to_row() -> dict[str, Any]` を持ち、Parquet 直書きに対応。`MAX_PLAYERS=4` 分の列 (`agent_{i}_name`, `agent_{i}_score` 等) を常に出力。
  - `recorder.write_records()` は `run_id` + `mode` 単位で Parquet 1 ファイル生成。
  - `loader.list_matches()` は `pl.scan_parquet(..., hive_partitioning=True)` で `mode` 列を自動認識。
  - `loader.load_replay()` は `payload["configuration"]` と `payload["steps"]` から `make(...)` で env を再現。

- **Patterns used**:
  - Immutable frozen dataclass (`AgentSpec`, `MatchSpec`, `MatchRecord`, `AgentTiming`)
  - `pathlib.Path` 徹底、CLI は typer + rich。
  - I/O 分離: ワーカーは bytes を返すだけ、FS 書き込みは main プロセス。
  - Pickle 制約: `resolve(name)` は worker 内で lazy import（`env.agents.agent_version()` は git SHA）。

- **Coupling & side effects**:
  - `MatchRecord.to_row()` は schema 固定（12列 + agent列×4×6）。新フィールド追加時はバージョン更新必須。
  - `recorder.write_records()` は rows を polars DataFrame 経由で書くため、dict キーの型整合が必要。
  - `loader.list_matches()` は `index.parquet/` 配下しか見ないため、別ルートを追加する場合は分析系の利用箇所を明確に分離。

- **Test coverage**: `tests/env/` に test_runner / test_recorder / test_loader の pytest が揃う（命名規則は `tests/<mirror>/test_*.py`）。kaggle_environments を直接 make するテストと、I/O ユニットテストが混在。

- **Gaps identified**:
  - 外部ソース（Kaggle 本番対戦ログ）のインポート経路がない。
  - `MatchRecord` に rating (μ/σ) や `source` 識別子が無い。
  - `replay_path` は `replays/{match_id}.json.gz` 固定。外部ソースでも流用可能な命名規則。

### Area 2: Kaggle CLI ラッパー (`src/submit/`)

- **Files analyzed**:
  - `src/submit/kaggle_api.py:1-184` — `kaggle competitions ...` の subprocess ラッパー。`submit`, `confirm_submission`, `list_submissions`, `poll`。
  - `src/submit/__main__.py` — typer CLI（submit/history/poll）。
  - `src/submit/history.py:1-44` — `data/submissions/<case>/submissions.jsonl` への追記。
  - `src/submit/packager.py` — `.submitignore` 対応のアーカイブ作成。
  - `src/submit/auth.py` — 認証ヘルパ（kaggle.json 認識）。

- **Current implementation**: すべて `subprocess.run(["kaggle", ...])` 経由。認証は `~/.kaggle/kaggle.json` / 環境変数に依存。`list_submissions()` は自身の提出履歴だけを取得する（他ユーザーは対象外）。

- **Key interfaces**:
  - `_run(args: list[str]) -> str` — 例外 `KaggleCLIError` で包む CLI 実行。
  - `list_submissions() -> list[dict[str, str]]` — CSV → dict 変換。

- **Coupling & side effects**:
  - `subprocess` 起動コストが高いので、ループ内呼び出しは避ける設計。
  - 他ユーザーのリーダーボード取得 (`kaggle competitions leaderboard`) は未実装だが、CLI 自体は `kaggle competitions leaderboard -c orbit-wars --show` で利用可。

- **Test coverage**: `tests/submit/` に subprocess をモックする形でユニットテストあり。

- **Gaps identified**:
  - リーダーボード取得 API の薄いラッパーが未実装。
  - Kaggle の内部 Episode Service (`https://www.kaggle.com/requests/EpisodeService/*`) は kaggle CLI には無く、`kaggle_environments.api` に別途存在する。

### Area 3: `kaggle_environments.api` の内蔵機能（外部パッケージ）

- **Files analyzed (external)**: `kaggle_environments/api.py`
  - `list_episodes(episode_ids: list[int]) -> dict` — `POST https://www.kaggle.com/requests/EpisodeService/ListEpisodes` / body `{"Ids": episode_ids}`。
  - `list_episodes_for_team(team_id: int) -> dict` — `{"TeamId": team_id}`。
  - `list_episodes_for_submission(submission_id: int) -> dict` — `{"SubmissionId": submission_id}`。
  - `get_episode_replay(episode_id: int) -> dict` — `POST .../GetEpisodeReplay` / body `{"EpisodeId": episode_id}`。認証なしで公開。

- **Implication**: 公式ライブラリにリプレイ取得関数が揃っているため、独自の HTTP 実装を作らずこれを呼び出せばよい。`kaggle_environments>=1.17.0` は既に依存として入っている。

### Area 4: データレイクの現状 (`data/`)

- `data/` 直下は `lake/` のみ（gitignore）。`data/matches/` は評価実行後に生成される。
- `.gitignore` でほぼ全て無視。parquet / json.gz は commit 対象外。
- CI は `dev/test-backend` で format→lint→mypy→pytest のみ、データ生成系は走らない。

## Technical Constraints

1. **Kaggle レート制限**: `requests/EpisodeService/*` は公開だが、参考実装（Halite スクレイパー）では **60 req/min** を守っている。違反で IP ベースの一時的 BAN 報告あり。
2. **リプレイサイズ**: Orbit Wars は 500 turn × planets/fleets 配列で JSON が大きくなる傾向。Halite で数百 KB–数 MB / episode。1,000 episode で 0.5–5 GB 規模になり得る。
3. **Kaggle CLI 依存**: リーダーボード取得は `kaggle competitions leaderboard -c orbit-wars --show` が必須。認証は `~/.kaggle/kaggle.json`。
4. **Parquet 書き込み**: polars 書き込みは一括で行う必要があり、append は run 単位でファイル分割する現行設計に従う。
5. **mode 識別**: Kaggle 側 episode は `agents` 数 (`len` が 2 or 4) で mode を推定できる。configuration に記録がある場合はそれを優先。
6. **Python 3.13 + mypy strict**: 新モジュールも `Any` 禁止、`type: ignore` 禁止、frozen dataclass 前提。
7. **ファイル行数**: 200-400 行が標準、800 行上限。新モジュール合計を 500 行以下で収める見込み。

## Key Findings Summary

- **公式関数を全面活用**: `kaggle_environments.api.{list_episodes, list_episodes_for_submission, get_episode_replay}` が揃っており、HTTP 層を自前実装する必要なし。
- **レートリミット必須**: 60 req/min を `time.sleep` ベースの簡易トークンバケットで守る。429 時は exponential backoff。
- **既存構造の踏襲**: `data/matches/` を模して `data/kaggle_episodes/` に `replays/{episode_id}.json.gz` + `index.parquet` を配置。ロジックは `src/env/kaggle/` を新設し、既存 `recorder.py` の関数を再利用可能な形で共用。
- **レジューム**: 取得済 `episode_id` を `index.parquet` から `pl.scan_parquet(...).select("episode_id").unique()` で参照し skip。状態ファイルを別途持つ必要なし。
- **メタデータスキーマ**: `MatchRecord` 互換を基本とし、`episode_id`, `submission_id`, `team_id`, `rating_mu`, `rating_sigma`, `source="kaggle"`, `scraped_at` を追加。`SCHEMA_VERSION` を 2 に上げる。
- **CLI 統合**: 既存 `python -m env` の typer app に `kaggle` サブコマンド群を追加、あるいは `python -m env.kaggle` として独立 CLI。本設計では後者を採用（責務分離 + 学習データ収集は評価系と役割が異なるため）。
