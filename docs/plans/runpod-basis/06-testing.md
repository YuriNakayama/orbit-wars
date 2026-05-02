# runpod-basis — Test Strategy

## Testing Approach

3 層に分けて品質を担保する (vast-ai-basis と同方針):

1. **Unit tests**: `backend/src/runpod_io/` の各モジュールを SDK / subprocess / GraphQL レベルで mock し、純粋 Python ロジック (offer 解析、テンプレ置換、run.json schema 拡張、cost 集計、volume CRUD) を完全カバー。`backend/pipeline/imitation/case{1,3,4}/training/train.py` の RunPod env 検出は CPU で deterministic に test。
2. **Integration tests (lite)**: `dev/test-backend` パイプラインに組み込み、`backend/src/runpod_io/` の cli → offers → instance → volumes 結合を `unittest.mock.patch` で stub しつつ end-to-end 実行。実 RunPod は呼ばない。
3. **Manual e2e (cost-aware)**: 実 RunPod pod 起動を **1 回のみ** 行い、本物の小規模 run で onstart → dvc push → ローカル pull → promote までを手動検証。初回成功後は `docs/plans/runpod-basis/README.md` に手順を記録。

カバレッジ目標は **`backend/src/runpod_io/` で 80%+、`backend/pipeline/imitation/case{1,3,4}/training/train.py` の既存カバレッジを下げない**。

## Unit Tests

### Backend (pytest)

- **`backend/tests/src/vast/test_run_meta.py`** (Step 1)
  - 既存テスト全 pass。
  - 新規: `RunMetadata` に `runpod_pod_id` / `runpod_offer_snapshot` を渡した round-trip。
  - 後方互換: 旧フィールド欠如 JSON を `read_run_json` で読み default `None` で埋まる。

- **`backend/tests/pipeline/imitation/case{1,3,4}/training/test_train_run_dir.py`** (Step 2)
  - 既存 vast 専用テスト (`ORBIT_WARS_VAST_INSTANCE_ID` のみ) は全 pass。
  - 新規 (各 case 同等):
    - `ORBIT_WARS_RUNPOD_POD_ID="abc"` のみ → run.json に `runpod_pod_id="abc"`、`vast_instance_id=None`。
    - 両方セット → `RuntimeError` (両 provider 排他)。
    - `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT='{"gpu_type_id":"X","dph_total":0.5}'` で snapshot dict 展開。
    - `ORBIT_WARS_RUN_DIR` 無し + `ORBIT_WARS_RUNPOD_POD_ID` set → assertion error (Risk #4 防御)。

- **`backend/tests/src/runpod_io/test_auth.py`** (Step 3)
  - `RUNPOD_API_KEY` 未設定で `CredentialsError` (actionable message に `runpod.io/console/user/settings` を含む)。
  - `.env` ファイルから読める (`tmp_path` + dotenv)。
  - 環境変数 fallback。
  - `load_aws_creds` の re-export が動く (vast.auth と同じ subprocess mock)。

- **`backend/tests/src/runpod_io/test_run_meta.py`** (Step 4)
  - re-export smoke: `from runpod_io.run_meta import RunMetadata, generate_run_id, hash_params, write_run_json` が動く。
  - `build_runpod_offer_snapshot()` の dict 構造。

- **`backend/tests/src/runpod_io/test_offers.py`** (Step 5)
  - `sdk.get_gpus` / `sdk.get_gpu` を mock、固定 dict で `Offer` 変換。
  - `cloud_type=SECURE` で community-only GPU が除外。
  - `cloud_type=ALL` で同じ GPU から 2 つの Offer (SECURE/COMMUNITY) が生成。
  - dph asc ソート。
  - `min_memory_gb` フィルタ。
  - `format_table()` が rich Table を生成。
  - `pick_offer()` の対話入力 (IntPrompt mock)。

- **`backend/tests/src/runpod_io/test_instance.py`** (Step 6)
  - `render_onstart()` が 9 placeholder すべてを置換。
  - shell injection 試行 (`; rm -rf /`) で `TemplateError`。
  - `_VALID_VALUE` / `_VALID_CONFIG_ARG` / `_VALID_PREPROCESS_CMD` の境界値テスト。
  - `build_env_dict()` の env 名 validation (`^[A-Z_][A-Z0-9_]*$`)。
  - `create_pod()` の SDK 呼び出し引数を mock で assert (`docker_args` の `bash -c` ラッピング、`network_volume_id`、`cloud_type`、`gpu_type_id`、`env`)。

- **`backend/tests/src/runpod_io/test_onstart_template.py`** (Step 6)
  - `bash -n backend/src/runpod_io/onstart.sh.tmpl` の syntax check (subprocess)。
  - placeholder を valid 値で置換した後の bash -n も pass。

- **`backend/tests/src/runpod_io/test_volumes.py`** (Step 7)
  - `run_graphql_query` を mock、固定応答で `Volume` / `VolumeOffer` 変換。
  - `find_volume_by_name` の name 一致と複数一致時の挙動 (最新 id を返す)。
  - `validate_volume_name` の制約。

- **`backend/tests/src/runpod_io/test_cost.py`** (Step 8)
  - 3 個の固定 run.json fixture (1 vast + 2 runpod) で、runpod 2 個のみ集計されること。
  - 月単位 filter の境界条件 (前月最終日 / 当月初日)。
  - markdown 出力構造 (table headers, total/adopted/average)。

- **`backend/tests/src/runpod_io/test_cli.py`** (Step 9-13)
  - **`test_cli_help`** (Step 9): `python -m runpod_io --help` で 5 サブコマンド (train/pull/promote/cost-report/volume) 表示 smoke。
  - **`test_train_*`** (Step 10):
    - mock 環境下の正常 path: `runpod_io.offers.search_offers` / `instance.create_pod` / `auth.*` を mock し、`create_pod` の引数を assert。
    - unpushed sha → `BadParameter`。
    - cost limit 超過 → `confirm` 経由 (yes/no 両分岐)。
    - `--cloud-type=COMMUNITY` で `network_volume_id=None` の挙動。
    - `--volume-id` 明示 / `--volume-name` 一致再利用 / `--auto-create-volume` の 3 path。
  - **`test_pull_*`** (Step 11): `subprocess.run` mock で正常/異常 path、status 警告。
  - **`test_promote_*`** (Step 12): cp + dvc commit の subprocess 呼び出し検証、`run.json.status="adopted"` 更新。
  - **`test_cost_report_*`** (Step 13): markdown 生成 path 検証。
  - **`test_volume_*`** (Step 13): typer wiring + SDK 引数。

### Frontend
- N/A (本 feature にフロントエンド変更なし)

## Integration Tests

### API tests
- N/A (外部 API は RunPod のみ、test では SDK / GraphQL を mock)

### CLI integration
- `tests/src/runpod_io/test_cli.py` で `typer.testing.CliRunner` を使い、`runpod train --help`, `runpod pull --help`, `runpod volume --help` 等のコマンド階層を検証。
- `dvc repro` を伴う slow integration test は **slow marker** で skip 可能化。`dev/test-backend` のデフォルトでは実行しない (vast 同等)。

### SDK 衝突回避の smoke test
- Step 3 の acceptance に **`python -c "import runpod_io; import runpod; print(runpod_io.__name__, runpod.__name__)"` が成功すること** を含める。これにより:
  - パッケージ名衝突がないこと。
  - 公式 SDK が期待通り import 可能であること。

## E2E Tests

実 RunPod 起動を含む E2E は **手動** で 1 度行い、その手順を docs に記録する方針:

1. **準備**:
   - `backend/.env` に `RUNPOD_API_KEY` 設定。
   - `~/.aws/credentials` の `orbit-wars` profile が利用可能。
   - RunPod Web UI で network volume 作成 (DC: US-KS-2, 15GB, name: `orbit_wars_cache_runpod`)。
2. **commit & push**: `feature/runpod-basis` ブランチで実装を commit & push。
3. **実行**: `dev/runpod train <sha> --case case1 --cloud-type SECURE --seed 0 --label e2e-test`。
4. **モニタ**: `runpodctl pod logs <id>` で onstart 進行を 10-15 分監視。
5. **検証ポイント**:
   - Pod が destroy されたか (`runpodctl pod list` で消えていること)。
   - DVC remote (`s3://orbit-wars-dvc-...`) に新 hash の object 増加。
   - `dev/runpod pull <run_id>` で best.pt + metrics.json + run.json が取得可能。
   - `run.json` に: `status="pushed"`, `runpod_pod_id` 非 None, `runpod_offer_snapshot.cloud_type="SECURE"`, `train_metrics.device="cuda"`, `gpu_name`。
   - `vast_instance_id` / `vast_offer_snapshot` は `None` (両 provider 排他確認)。
   - 推定コスト < 0.5 USD (Secure RTX 3090 0.5h)。
6. **promote 検証**: `dev/runpod promote <run_id>` で `policy/weights.pt` 更新、`git status` 表示。
7. **cost-report**: `dev/runpod cost-report --month <YYYY-MM>` で markdown 生成、内容に当該 run が含まれること。

E2E 結果は `docs/experiment/<run_id>.md` に記録し、PR description にリンク。

## Test Data

- **mock の参照 dict**:
  - `backend/tests/src/runpod_io/fixtures/sample_gpus.json`: `runpod.get_gpus()` の固定応答。
  - `backend/tests/src/runpod_io/fixtures/sample_gpu_detail.json`: `runpod.get_gpu(id)` の固定応答 (price 情報含む)。
  - `backend/tests/src/runpod_io/fixtures/sample_create_pod_response.json`: `runpod.create_pod()` の固定応答。
- **run.json fixtures**:
  - `backend/tests/src/runpod_io/fixtures/runs/<dummy_runpod_run>/run.json` (runpod_offer_snapshot あり)。
  - `backend/tests/src/runpod_io/fixtures/runs/<dummy_vast_run>/run.json` (vast_offer_snapshot あり、runpod 集計から除外されることの確認用)。
- **既存 train データ**: `data/mart/imitation/case1/{train,val}.parquet` (DVC pull 済前提)。slow テストでなければ tiny synthetic dataset を別 fixture で用意。

## Coverage Targets

- **`backend/src/runpod_io/`**: 80%+
  - `cli.py` の subcommand wiring と分岐
  - `offers.py` の SDK ラッパ + format/pick + cloud_type 分岐
  - `instance.py` の render + create_pod + sanitize
  - `run_meta.py` の re-export smoke
  - `cost.py` の aggregator + filter (vast skip) + markdown
  - `volumes.py` の CRUD + GraphQL mock
  - `auth.py` の credentials load
- **`backend/pipeline/imitation/case{1,3,4}/training/train.py`**: 既存カバレッジ維持 (regression 防止)
- **`backend/src/vast/run_meta.py`**: 新フィールド追加分のテスト追加で 100% (default なので分岐ゼロ)
- **`backend/src/runpod_io/onstart.sh.tmpl`**: bash -n syntax check のみ (実行カバレッジは E2E で確認)

## Continuous Integration

- 既存 `dev/test-backend` (`format → lint → type → pytest`) に **新規 `runpod_io/` パッケージのテストを追加**。
- mypy で `runpod_io/` を型チェック (typer の Annotated パラメータ + dataclass)。`runpod` SDK の型 stub が無い場合 `# type: ignore[import-untyped]` で凌ぐ。
- ruff で `runpod_io/` の style 統一。
- 実 RunPod を含む E2E test は `pytest -m runpod_e2e` などで marker 制御し、CI ではデフォルト skip。

## Cross-Provider 回帰テスト

両基盤共存ゆえ、Vast 基盤の既存テストが壊れていないことを必ず検証:

- `backend/tests/src/vast/` 全テスト pass (Step 1 で `RunMetadata` 拡張時に特に注意)。
- vast 既存 run.json fixtures の読み込みが default 値で埋まり、既存ロジックが動くこと。
- `dev/test-backend` で format/lint/type/pytest が両パッケージ含めグリーン。
