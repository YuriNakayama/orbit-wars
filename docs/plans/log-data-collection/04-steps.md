# Log Data Collection — Implementation Steps

Backend-first の順序で 8 ステップに分解。各ステップは独立した PR で完結可能。Step 1–3 は並列実行可能、以降は前段に依存する。

## Step 1: スキーマ v2 拡張

**Target**: backend
**Dependencies**: None

### Overview
`MatchRecord` に Kaggle ソース向けフィールドを追加し、`SCHEMA_VERSION=2` へ更新。selfplay レコードとの互換性を保ちつつ、新列にデフォルト値を提供する。

### Work Items
- [ ] `src/env/types.py` の `SCHEMA_VERSION = 2` へ更新。
- [ ] `MatchRecord` に `source: str = "selfplay"`, `episode_id: int = -1`, `scraped_at: str = ""` 追加。
- [ ] 各 agent 用フィールド `agent_{i}_submission_id`, `agent_{i}_team_id`, `agent_{i}_rating_mu`, `agent_{i}_rating_sigma`, `agent_{i}_state` を `to_row()` / `from_row()` に追加。
- [ ] 既存 `selfplay` 由来の呼び出しでデフォルト埋めが動くか確認（`run_one_match` 等）。
- [ ] 既存テスト (`tests/env/test_types.py` 等) を更新。

### Target Files (Expected)
- `src/env/types.py`
- `tests/env/test_types.py`
- `src/env/executor.py`（record dict への新フィールド追加）

### Acceptance Criteria
- `ruff`, `mypy`, `pytest` が全通。
- 既存 selfplay run を実行しても Parquet 書き出しに失敗しない。
- 新列が全て `default_factory` またはデフォルト値で埋まる。

---

## Step 2: HTTP クライアント + レートリミッタ

**Target**: backend
**Dependencies**: None

### Overview
Kaggle EpisodeService を叩く薄い HTTP ラッパーと、60 req/60 s の token bucket を実装する。

### Work Items
- [ ] `src/env/kaggle/__init__.py`, `client.py`, `rate_limit.py`, `types.py` を新規作成。
- [ ] `build_session(config)` で `requests.Session` + `urllib3.Retry` を組み立て。
- [ ] `_post`, `list_episodes_for_team`, `get_episode_replay`, `extract_replay_json` を実装。
- [ ] `TokenBucket(capacity=60, window_sec=60.0)` をスレッドセーフに実装。
- [ ] `KaggleEpisodeError` を定義し、`RequestException` を chain。
- [ ] 単体テスト: `responses` or `requests_mock` を dev 依存に追加 or unittest.mock で注入。

### Target Files (Expected)
- `src/env/kaggle/__init__.py`
- `src/env/kaggle/client.py`
- `src/env/kaggle/rate_limit.py`
- `src/env/kaggle/types.py`
- `tests/env/kaggle/test_client.py`
- `tests/env/kaggle/test_rate_limit.py`

### Acceptance Criteria
- `tests/env/kaggle/test_rate_limit.py::test_bucket_waits_when_full` が passes。
- `tests/env/kaggle/test_client.py::test_get_episode_replay_success/http_error/retry` が passes。
- レスポンスが `{"result": {"replay": "..."}}` 形式でなければ `KaggleEpisodeError`。

---

## Step 3: リーダーボード取得

**Target**: backend
**Dependencies**: None

### Overview
`kaggle competitions leaderboard -c orbit-wars --show --csv` の stdout を parse し、top N の `TeamRank` リストを返す。

### Work Items
- [ ] `src/env/kaggle/leaderboard.py` を新規作成。
- [ ] subprocess ラッパー `fetch(top, raw_snapshot=None)`。
- [ ] CSV parse（`teamId`, `teamName`, `score`, `submissionDate`）。
- [ ] `raw_snapshot` が指定された場合は CSV を保存。
- [ ] kaggle CLI が無い環境向けに `FileNotFoundError` を `KaggleCLIError` で包む（`src/submit/kaggle_api.py` と整合）。
- [ ] 単体テスト: subprocess を `monkeypatch` でモック。

### Target Files (Expected)
- `src/env/kaggle/leaderboard.py`
- `tests/env/kaggle/test_leaderboard.py`

### Acceptance Criteria
- fixtures の CSV から `top=5` で 5 行の `TeamRank` が返る。
- CLI 非存在環境で適切なエラー型が raise される。
- `raw_snapshot` 指定時にファイル書き込み確認。

---

## Step 4: Resume 用 State 抽出

**Target**: backend
**Dependencies**: Step 1

### Overview
既存 Parquet index から `episode_id` unique set を抽出し、再取得を抑止する。

### Work Items
- [ ] `src/env/kaggle/state.py` を新規作成。
- [ ] `existing_episode_ids(data_root, modes=None) -> set[int]` を実装（`pl.scan_parquet` + hive_partitioning=True）。
- [ ] Parquet 未作成時は空 set を返す。
- [ ] `source == "kaggle"` の行のみ対象（selfplay は除外）。
- [ ] 単体テスト: fixture Parquet から set を抽出。

### Target Files (Expected)
- `src/env/kaggle/state.py`
- `tests/env/kaggle/test_state.py`

### Acceptance Criteria
- 空ディレクトリで `set()` を返す。
- mode フィルタが正しく動く。
- selfplay 行は除外される。

---

## Step 5: Record 変換ロジック

**Target**: backend
**Dependencies**: Step 1, Step 2

### Overview
Kaggle EpisodeService のレスポンスから `MatchRecord`（v2）を生成する。mode 判定、勝敗判定、rating 抽出を含む。

### Work Items
- [ ] `src/env/kaggle/records.py` に `infer_mode`, `build_match_record`, `_resolve_outcome` を実装。
- [ ] `EpisodeMeta` dataclass を `types.py` に追加。
- [ ] mode 判定: `len(agents)` で `{2: "1v1", 4: "ffa4"}`。対応外は `ValueError`。
- [ ] 勝敗: `state == "ACTIVE"` が 1 人 → 勝者、複数同点 → `draw=True`、スコア最高で tie break。
- [ ] 日付 parse: `createTime`/`endTime` ISO 8601 → `elapsed_sec`。
- [ ] 単体テスト: 代表的な 1v1 / ffa4 / TIMEOUT レスポンスで期待 record を生成。

### Target Files (Expected)
- `src/env/kaggle/records.py`
- `src/env/kaggle/types.py`（`EpisodeMeta` 追記）
- `tests/env/kaggle/test_records.py`

### Acceptance Criteria
- 1v1 正常ケース、ffa4 正常ケース、TIMEOUT ケース、draw ケースで `build_match_record` が期待通り。
- 既存 `MatchRecord.to_row()` が新フィールド込みで正しく dict 化。

---

## Step 6: Recorder 汎用化 + Loader 拡張

**Target**: backend (cross-cutting)
**Dependencies**: Step 1

### Overview
`src/env/recorder.py` を `data_root` 下の任意 prefix（例 `data/kaggle_episodes/matches/`）に書き込めるように汎用化。`loader.py` に `load_kaggle_replay(episode_id)` を追加。

### Work Items
- [ ] `recorder.write_records` / `write_replay` / `write_run` のシグネチャはそのまま（`data_root` を基点に `matches/` 以下に書く）。新規要件は `data_root = Path("data/kaggle_episodes")` を渡すだけ。
- [ ] ただし `replay_path` の組み立てが `replays/{match_id}.json.gz` 固定なので、`MatchRecord.replay_path` を尊重する形に変更（既に record 側で `replays/{episode_id}.json.gz` を指定済 → 整合）。
- [ ] `loader.load_replay_payload(match_id=None, episode_id=None, data_root)` へ拡張するか、`load_kaggle_replay(episode_id, data_root)` を別関数で追加（後者を推奨、疎結合を維持）。
- [ ] 単体テスト: `tests/env/test_recorder.py` に `data_root=tmp_path/"kaggle_episodes"` ケースを追加。

### Target Files (Expected)
- `src/env/recorder.py`
- `src/env/loader.py`
- `tests/env/test_recorder.py`
- `tests/env/test_loader.py`

### Acceptance Criteria
- `data/matches/` と `data/kaggle_episodes/matches/` 両方で書き込み/読み込みが通る。
- 既存テストが全 passes。

---

## Step 7: Scraper Orchestration

**Target**: backend
**Dependencies**: Step 2, 3, 4, 5, 6

### Overview
`ScrapeSpec` を受け取り、leaderboard 取得 → team ループ → episode ループ → records/replay 生成 → 永続化の全体フローを実装する。

### Work Items
- [ ] `src/env/kaggle/scraper.py` に `ScrapeSpec`, `ScrapeResult`, `run(spec, progress=None)` を実装。
- [ ] `rich.Progress` で進捗表示（team x episode の2段）。
- [ ] 既取得 id を `state.existing_episode_ids(...)` で取得、in-memory set で run 内重複も防止。
- [ ] `--dry-run` は API 呼び出しは行うが gzip 書き出しと Parquet 書き出しを skip、summary dict のみ返す。
- [ ] `--limit-per-team` で team あたり取得 episode 数を制限。
- [ ] `KeyboardInterrupt` ハンドラ: バッファ済の record/replay を flush してから raise。
- [ ] `include_failed=False` のときは `state in {"ERROR", "INVALID"}` を skip。
- [ ] 単体テスト: mocked client/leaderboard で scraper を 1v1/ffa4 混合シナリオで実行。

### Target Files (Expected)
- `src/env/kaggle/scraper.py`
- `tests/env/kaggle/test_scraper.py`

### Acceptance Criteria
- 既取得 id を skip する。
- Dry-run で FS に書かない。
- mode フィルタ反映。
- KeyboardInterrupt で flush される。

---

## Step 8: CLI (`env.kaggle`)

**Target**: backend (CLI)
**Dependencies**: Step 7

### Overview
`python -m env.kaggle scrape|list|inspect` の Typer CLI を実装、summary を rich.Table で表示。

### Work Items
- [ ] `src/env/kaggle/__main__.py` と `src/env/kaggle/cli.py` を新規作成。
- [ ] `scrape` サブコマンド: `--top`, `--modes`, `--limit-per-team`, `--dry-run`, `--include-failed`, `--data-root`。
- [ ] `list` サブコマンド: 既存 `env list` と同様、Kaggle 側 index を表示。`--mode`, `--limit`, `--data-root`。
- [ ] `inspect` サブコマンド: 単一 `episode_id` のメタ（turns, winner, agent rating）を表示。
- [ ] `report.py` を re-use（summary table）。
- [ ] pyproject.toml の ruff per-file-ignores に `src/env/kaggle/cli.py = B008` を追加。
- [ ] 統合テスト: typer CliRunner で `scrape --dry-run --top 1` が正常終了。

### Target Files (Expected)
- `src/env/kaggle/__main__.py`
- `src/env/kaggle/cli.py`
- `pyproject.toml`（per-file-ignores 追加）
- `tests/env/kaggle/test_cli.py`

### Acceptance Criteria
- `uv run python -m env.kaggle scrape --dry-run --top 1` が exit 0。
- `uv run python -m env.kaggle list --mode 1v1 --data-root data/kaggle_episodes` が Parquet 空で "no matches found" を出力。
- `uv run python -m env.kaggle --help` で 3 サブコマンドが見える。
- `dev/test-backend` 全体 passes。

---

## 並列化可能性

- **並列可**: Step 1 / 2 / 3 は独立して着手可能。
- **逐次**: Step 4–5 は Step 1 依存、Step 6 は Step 1 依存。Step 7 は 2/3/4/5/6 全てに依存。Step 8 は Step 7 依存。

## クロスカッティング

- **ドキュメント**: 各 Step 完了時に `README.md` に `uv run python -m env.kaggle scrape` の例を追記。最終 Step で `docs/plans/log-data-collection/` へのリンクを README から張る。
- **Lint/Format**: 各 Step の PR で `dev/format` / `dev/lint` 必須。
- **CI**: `dev/test-backend` は既存のまま。新モジュールのテストはそれに乗る。
