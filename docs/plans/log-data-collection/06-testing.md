# Test Strategy

## Testing Approach

外部 HTTP と Kaggle CLI を完全にモックし、`src/env/kaggle/` をユニット中心でカバーする。Parquet の読み書きは `tmp_path` fixture で実データを使い、IO バグを早期検出する。スマートに絞り込み:

- **Fast unit tests (CI 必須)**: HTTP response, CLI output, Record 変換、Parquet IO
- **Local integration (手動)**: 実 Kaggle API を 1 episode だけ取得する `--live` マーカー
- **No E2E UI**: 本件は UI なし

カバレッジ目標: 80%+（`backend.md` 準拠）。

## Unit Tests

### `tests/env/kaggle/test_client.py`
- `test_build_session_has_retry_adapter`: mount 済の adapter が 429 を含むこと
- `test_list_episodes_for_team_success`: `{"result": {"episodes": [...]}}` が返る
- `test_get_episode_replay_success`: replay 文字列が dict 経由で取れる
- `test_extract_replay_json_missing_key_raises`: `result.replay` 欠損時に `KaggleEpisodeError`
- `test_post_http_error_raises`: 503 など raise_for_status でエラー
- `test_post_connection_error_chained`: `RequestException` が `KaggleEpisodeError` にチェーン
- モックには `responses` もしくは `unittest.mock` で `Session.post` をパッチ。依存最小化のため `unittest.mock` を優先

### `tests/env/kaggle/test_rate_limit.py`
- `test_bucket_allows_up_to_capacity_immediately`
- `test_bucket_waits_when_full`: 61 req 目で sleep が発生することを `time.sleep` モックで確認
- `test_bucket_threadsafe`: threading.Thread で並列 acquire が安全

### `tests/env/kaggle/test_leaderboard.py`
- `test_fetch_parses_csv`: fixtures/leaderboard.csv を stdout 返す subprocess モックで `TeamRank` x N
- `test_fetch_top_limits_rows`
- `test_fetch_raw_snapshot_writes_file`
- `test_fetch_cli_missing_raises_kaggle_cli_error`

### `tests/env/kaggle/test_state.py`
- `test_existing_ids_empty_dir`: set() を返す
- `test_existing_ids_from_index_parquet`: fixture Parquet で set 抽出
- `test_existing_ids_filters_by_mode`
- `test_existing_ids_excludes_selfplay`: `source != "kaggle"` は除外

### `tests/env/kaggle/test_records.py`
- `test_infer_mode_1v1`, `test_infer_mode_ffa4`, `test_infer_mode_invalid_raises`
- `test_build_match_record_1v1_winner`
- `test_build_match_record_ffa4_draw`
- `test_build_match_record_timeout_state`: `agent_states` に `TIMEOUT` が入る
- `test_build_match_record_to_row_roundtrip`: record → row → record で同値

### `tests/env/kaggle/test_scraper.py`
- `test_run_skips_existing_ids`
- `test_run_respects_mode_filter`
- `test_run_dry_run_does_not_write_fs`: tmp_path を使い何も書かれていないこと
- `test_run_handles_episode_error_and_continues`: 1 件失敗しても他 episode が取得される
- `test_run_include_failed_flag`
- `test_run_limit_per_team`
- `test_run_keyboard_interrupt_flushes_buffered`: `side_effect=KeyboardInterrupt` で途中中断

### `tests/env/kaggle/test_cli.py`
- `test_scrape_dry_run_exit_zero` (typer CliRunner + mocked scraper.run)
- `test_list_empty_shows_no_matches`
- `test_inspect_unknown_episode_exits_nonzero`
- `test_scrape_help_lists_options`

### `tests/env/test_types.py`（既存更新）
- `test_match_record_to_row_schema_version_2`
- `test_from_row_selfplay_backcompat`: 旧形式 dict でも `source` デフォルトで復元
- `test_from_row_kaggle_fields_roundtrip`

### `tests/env/test_recorder.py`（既存更新）
- `test_write_run_alternate_data_root`: `tmp_path/"kaggle_episodes"` で書ける
- `test_replay_filename_uses_match_id`: `replays/kaggle_ep_123.json.gz` 相当

### `tests/env/test_loader.py`（既存更新）
- `test_load_kaggle_replay_roundtrip`: `data_root=tmp/"kaggle_episodes"` で Round-trip

## Integration Tests

### `tests/env/kaggle/test_integration.py`（`@pytest.mark.live`, CI では skip）
- `test_live_single_episode`: 実 API で 1 episode 取得、JSON が kaggle_environments で make 可能
- `test_live_leaderboard`: `kaggle` CLI 実行で top 3 取得
- 実行条件: `RUN_LIVE_KAGGLE_TESTS=1` 環境変数、かつ `~/.kaggle/kaggle.json` 存在

## Test Data

### Fixtures
- `tests/env/kaggle/fixtures/leaderboard.csv` — 10 行サンプル（`teamId`, `teamName`, `score`, `submissionDate`）
- `tests/env/kaggle/fixtures/list_episodes_response.json` — `ListEpisodes` 代表レスポンス（1v1 × 2, ffa4 × 1）
- `tests/env/kaggle/fixtures/get_episode_replay_response.json` — `GetEpisodeReplay` 代表レスポンス（`result.replay` は小さい steps の JSON 文字列）
- `tests/env/kaggle/fixtures/timeout_response.json` — TIMEOUT agent 入り
- `tests/env/kaggle/fixtures/selfplay_index.parquet` — selfplay 1 件 + kaggle 1 件の Parquet

### Builder Helpers
- `make_episode(mode, agents_count=2, states=("ACTIVE", "INACTIVE"))` — fixtures とパラメトライズ両対応のビルダ関数

## Coverage Targets

- Unit: **80%+** (backend.md 準拠)
- `src/env/kaggle/client.py`: 90%+
- `src/env/kaggle/rate_limit.py`: 95%+
- `src/env/kaggle/records.py`: 90%+
- `src/env/kaggle/scraper.py`: 80%+（orchestration は条件分岐が多いため）
- `src/env/kaggle/cli.py`: 70%+（CliRunner での入出力は手厚くしすぎない）

## CI Hooks

- `dev/test-backend` (既存) に新モジュールのテストが自動で乗る（`pytest tests/` が `testpaths` で解決）
- Live 実 API は CI で走らない（`pytest -m "not live"` を既定にする）
- 新依存 (`responses` 等) は `[dependency-groups] lint` or `env` 追加を検討（最小化のため `unittest.mock` 優先）

## Performance Validation

単体ではなく、最初の手動 live run で以下を測定:
- 30 team × 平均 10 episode = 300 episode 取得の総時間（期待値 10 分以内）
- Parquet 書き込みサイズ・Index スキャン時間
- 60 req/min の実際の到達上限（429 が出ないこと）
