# vast-ai-basis — Test Strategy

## Testing Approach

3 層に分けて品質を担保する:

1. **Unit tests**: `backend/src/vast/` 内の各モジュールを subprocess / SDK レベルで mock し、純粋 Python ロジック（offer 解析、テンプレ置換、run.json schema、コスト集計）を完全カバー。`backend/pipeline/imitation/case1/training/train.py` の env override 経路は CPU 環境で deterministic にテスト。
2. **Integration tests (lite)**: `dev/test-backend` パイプラインに組み込む形で、`backend/src/vast/` のモジュール間結合（cli → offers → instance）を `unittest.mock.patch` で stub しつつ end-to-end で実行。実 Vast は呼ばない。
3. **Manual e2e (cost-aware)**: 実 Vast 起動を 1 回行い、本物の小規模 run で onstart → dvc push → ローカル pull → promote までを手動検証。**初回成功後は再現性のための手順書として `docs/plans/vast-ai-basis/README.md` (任意) に記録**、自動化は CI コスト的に見合わないため行わない。

カバレッジ目標は **`backend/src/vast/` で 80%+、`backend/pipeline/imitation/case1/training/train.py` は既存カバレッジを下げない**。Step 7 の決定で確定済み。

## Unit Tests

### Backend (pytest)

- **`backend/tests/vast/test_run_meta.py`** (Step 3 で作成)
  - `generate_run_id` のフォーマット正規表現マッチ
  - branch slug の `/` → `-` 置換 (e.g., `feature/vast-ai-basis` → `feature-vast-ai-basis`)
  - `hash_params` の dict 順序非依存性 (key を入れ替えても同じ hash)
  - `write_run_json` / `update_run_json` の round-trip + atomic rename
  - `RunMetadata` の schema_version=1 検証
- **`backend/tests/vast/test_auth.py`** (Step 5)
  - `aws configure get` を `subprocess.run` で mock し AwsCreds dataclass を返すケース
  - `VAST_API_KEY` 未設定時の `RuntimeError` で actionable message
- **`backend/tests/vast/test_offers.py`** (Step 6)
  - `VastAI().search_offers()` を mock、固定 dict を返して Offer dataclass 変換
  - dph asc ソート結果のテスト
  - `format_table` が rich Table を生成すること
  - `pick_offer` の対話入力 (rich Prompt を mock)
- **`backend/tests/vast/test_instance.py`** (Step 7)
  - `render_onstart` が 5 placeholder すべてを置換すること
  - shell injection 試行 (e.g., `; rm -rf /`) で `ValueError`
  - `build_env_string` の format 検証
  - `create_instance` の SDK 呼び出し引数の検証 (mock 経由)
- **`backend/tests/vast/test_onstart_template.py`** (Step 7)
  - `bash -n` で `onstart.sh.tmpl` の syntax check (subprocess)
  - placeholder を valid 値で置換した後の bash -n も pass
- **`backend/tests/vast/test_cli.py`** (Step 8)
  - `python -m vast --help` で 4 サブコマンド表示の smoke test (CliRunner)
- **`backend/tests/vast/test_cli_train.py`** (Step 9)
  - mock 環境下で正常 path
  - unpushed sha → fail-fast
  - cost limit 超過 → confirm prompt 経由
- **`backend/tests/vast/test_cli_pull.py`** (Step 10)
  - dvc pull の subprocess 引数検証
  - run.json status が `pushed` 以外で警告
- **`backend/tests/vast/test_cli_promote.py`** (Step 11)
  - cp と dvc commit の呼び出し検証
  - run.json status の `adopted` への更新
- **`backend/tests/vast/test_cost.py`** (Step 12)
  - 3 個の固定 run.json fixture で集計
  - 月単位フィルタの境界条件
  - markdown 出力の構造
- **`backend/tests/pipeline/imitation/case1/training/test_train_run_dir.py`** (Step 2 / Step 4)
  - `ORBIT_WARS_RUN_DIR` env 設定時に `run dir/{best.pt, metrics.json, run.json}` 3 ファイル生成
  - env 未設定時に従来通り `policy/weights.pt` のみ更新（regression）
  - `run.json` の schema_version=1 と必須フィールド (Step 4 で追加 assert)
  - `ORBIT_WARS_VAST_INSTANCE_ID` あり + `ORBIT_WARS_RUN_DIR` 無し → assertion error (Risk #4 防御)

### Frontend
- N/A（本 feature にフロントエンド変更なし）

## Integration Tests

### API tests
- N/A（外部 API は Vast.ai のみ、test では SDK を mock）

### CLI integration
- `tests/vast/test_cli.py` で `typer.testing.CliRunner` を使い、`vast train --help`, `vast pull --help` 等のコマンド階層を検証。
- `dvc repro train_imitation_case1` を `ORBIT_WARS_RUN_DIR=/tmp/test_run_dir` 付きで実行する `pytest` 経路は **slow marker** で skip 可能化。`dev/test-backend` のデフォルトでは実行しない（既存 slow テストと同パターン）。

## E2E Tests

実 Vast 起動を含む E2E は **手動** で 1 度行い、その手順を docs に記録する方針:

1. **準備**: `backend/.env` に `VAST_API_KEY` 設定、`~/.aws/credentials` の `orbit-wars` profile が利用可能であること
2. **commit**: `feature/vast-ai-basis` ブランチで実装を commit & push
3. **実行**: `dev/vast-train <sha> --stage train_imitation_case1 --seed 0 --label e2e-test`
4. **モニタ**: `vastai logs <id>` で onstart 進行を監視 (10〜15 分)
5. **検証ポイント**:
   - インスタンスが destroy されたか (`vastai show instances` で消えていること)
   - DVC remote (`s3://orbit-wars-dvc-286854171013/remote`) に新しい hash の object が増えたか
   - `dev/vast-pull <run_id>` で best.pt + metrics.json + run.json が取得できるか
   - `run.json` の `status=pushed`、`gpu_name`、`train_metrics.device=cuda` が記録されているか
   - 推定コスト < 0.5 USD であること（dph_total 0.13-0.29 × 0.5h）
6. **promote 検証**: `dev/vast-promote <run_id>` で `policy/weights.pt` が更新されること、`git status` に変更が出ること

E2E 結果は `docs/experiment/<run_id>.md` に記録し、PR description にリンク。

## Test Data

- **mock の参照 dict**: `backend/tests/vast/fixtures/sample_offers.json` で固定の search_offers 応答を保持
- **run.json fixtures**: `backend/tests/vast/fixtures/runs/<dummy_run_id>/run.json` を 3 個配置（cost.py のテスト用）
- **既存 train データ**: `data/mart/imitation/case1/{train,val}.parquet` (DVC pull 済前提) を Step 2 / Step 4 のテストでも使用。slow テストでなければ tiny synthetic dataset を別 fixture で用意

## Coverage Targets

- **`backend/src/vast/`**: 80%+
  - `cli.py` の subcommand wiring と分岐
  - `offers.py` の SDK ラッパ + format/pick
  - `instance.py` の render + create + sanitize
  - `run_meta.py` の dataclass + I/O + slug
  - `cost.py` の aggregator + markdown
  - `auth.py` の credentials load
- **`backend/pipeline/imitation/case1/training/train.py`**: 既存カバレッジ維持 (regression 防止)
- **`backend/src/vast/onstart.sh.tmpl`**: bash -n syntax check のみ

## Continuous Integration

- 既存 `dev/test-backend` の流れ (`format → lint → type → pytest`) に **新規 vast/ パッケージのテストを追加**
- mypy で `vast/` パッケージを型チェック (typer の Annotated パラメータ + dataclass)
- ruff で `vast/` パッケージの style 統一
- 実 Vast 起動を含む test は `pytest -m vast_e2e` などで marker 制御し、CI ではデフォルト skip
