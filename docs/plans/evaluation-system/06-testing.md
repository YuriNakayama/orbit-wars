# Evaluation System — Test Strategy

## 1. Testing Approach

- **unit テストは kaggle_environments をモック** or 呼ばない形で高速に回す。
- **integration テストは `@pytest.mark.integration` + `pytest.importorskip("kaggle_environments")`** で実環境を使い、最小エピソードで挙動を確認。
- **非決定性対策**: integration テストでは「勝者」「長さ」など seed 依存の値をアサートしない。`return type` / `レコード件数` / `ファイル存在` / `JSON round-trip` など**seed に依存しない性質のみ検証する**（リスク #2 対応）。
- **カバレッジ**: ライン 80%+ を目標。branch も緑を目指すが厳格 SLA ではない。
- `@pytest.mark.slow` は end-to-end テストのみに付与。ci-backend.yml の将来的な分離ジョブ用予約。

## 2. Unit Tests（Pytest、kaggle_environments 不要）

### 対象モジュール
- `src/env/types.py` → `tests/env/test_types.py`
- `src/env/agents.py` → `tests/env/test_agents.py`
- `src/env/recorder.py` → `tests/env/test_recorder.py`
- `src/env/loader.py`（Parquet スキャン部分のみ） → `tests/env/test_loader.py`
- `src/env/analyze.py` → `tests/env/test_analyze.py`
- `src/env/report.py` → `tests/env/test_report.py`

### 検証観点
| モジュール | 主要ケース |
|---|---|
| types | `MatchRecord.to_row` / `from_row` のラウンドトリップ、frozen 性、4人固定列の空文字埋め |
| agents | `resolve` の3分岐（module:attr / 文字列 / unknown→KeyError）、`agent_version()` が空文字を許容 |
| recorder | hive partition 書き込み、同一 run_id のサフィックス挙動、gzip bytes の round-trip |
| loader | 合成 parquet からの `list_matches`、filter push-down |
| analyze | モック 10 records で `agent_winrate` 期待値一致 |
| report | rich.Table の列 / 値チェック（セル単位） |

## 3. Integration Tests（kaggle_environments 使用）

### 対象
- `src/env/executor.py` → `tests/env/test_executor.py`
  - 1v1 で `case0` 同士 1 エピソードを実行、`turns > 0` / `winner in {-1,0,1}` / リプレイ bytes が decompress 可能
  - `save_replay=False` で bytes が None
- `src/env/runner.py` → `tests/env/test_runner.py`
  - `parallel=1` と `parallel=2` で各 2 エピソード → 両方 2 件返る
  - mode 不一致は `ValueError`（unit）
- `src/env/cli.py` → `tests/env/test_cli.py`
  - `CliRunner` で `run --agents case0,case0 --mode 1v1 -n 1 --parallel 1 --no-save-replay` → exit_code 0、stdout に "Summary"
- `src/env/loader.py`（実リプレイ）→ `tests/env/test_loader.py`
  - `run` で作ったリプレイを `load_replay` で読み、`env.render("json")` が dict

### 実行ポリシー
- すべて `@pytest.mark.integration` を付与。
- 冒頭に `kaggle_environments = pytest.importorskip("kaggle_environments")` を置き、ローカル未インストール環境でスキップ。
- エピソード数は各テスト 1-2 に抑え、30 秒以内に終わらせる。

## 4. End-to-End Tests

- `tests/env/test_end_to_end.py`
- `@pytest.mark.integration @pytest.mark.slow`
- シナリオ:
  1. `CliRunner` で `run --agents case0,case0 --mode 1v1 -n 2 --parallel 2 --save-replay` 実行
  2. `data/matches/index.parquet` が生成されていること
  3. `list_matches()` で 2 件返ること
  4. 1 件の `replay_path` をロードして `env.render("json")` が `steps` を含むこと
- 許容所要時間: 60 秒以内（8 コア開発機）

## 5. Snapshot / Regression Tests

- 既存 `tests/pipeline/case1/test_baseline_agent.py` のスナップショットを**絶対に壊さない**。
- `src/env/` のテストは snapshot 方式を採用しない（非決定性があるため）。代わりに合成データでロジックのみ検証する。
- `pipeline/case1/evaluation/snapshot_update.py` は変更・移動しない。

## 6. Test Data / Fixtures

- `tests/env/conftest.py` に以下の fixture を配置:
  - `tmp_data_root(tmp_path)` — `tmp_path / "matches"` を返す。各テスト後 tmp_path は自動削除
  - `sample_match_record()` — MatchRecord の合成インスタンス（4人記録）
  - `sample_records_mixed(n=10)` — agent_winrate / timing_distribution テスト用の合成 records
  - `orbit_wars_available` — `importlib.util.find_spec("kaggle_environments")` の boolean（integration 用）

## 7. CI 統合

- `dev/test-backend` はルート実行に修正済みであること（Step 0）。
- pyproject の `[tool.pytest.ini_options]` マーカー登録に `integration`, `slow` 両方すでに存在 → 追加作業不要。
- CI 実行パターン（ci-backend.yml 側は本件スコープ外、ただし推奨）:
  - `uv run pytest tests -m "not slow"` を default job で走らせる
  - `uv run pytest tests -m slow` を optional job で走らせる
  - integration ジョブは kaggle_environments が install されている前提

## 8. Coverage Targets

- Unit カバレッジ: **80%+（必須）**
- Branch カバレッジ: 70%+ を目指す（厳格 SLA ではない）
- Integration カバレッジ: 定量目標なし（機能ユースケースがカバーされていればよい）
- E2E シナリオ数: 最低 1 シナリオ（`run → list → load → render`）

## 9. Test Enumeration Summary

| 層 | テスト件数目安 | 所要時間目安 |
|---|---|---|
| Unit | 30〜40 件 | <2 秒 |
| Integration (non-slow) | 5〜8 件 | 10〜30 秒 |
| E2E (slow) | 1〜2 件 | 30〜60 秒 |
| 合計 | 〜50 件 | ~1 分 |

## 10. 手動検証（自動化対象外）

- `pipeline/case1/eda/replay_viewer.py` を VS Code / Jupyter で開き、セル実行 → 惑星描画が表示されること
- `uv run python -m env run --agents baseline_v1,case0 --mode 1v1 -n 10 --parallel 4 --save-replay` を実機で走らせ、rich.Table 出力と parquet / replays の生成を目視確認
