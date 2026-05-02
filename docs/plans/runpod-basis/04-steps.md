# runpod-basis — Implementation Steps

実装は **vast.run_meta の拡張 → train.py 改修 → runpod_io パッケージ → onstart テンプレ → CLI 4 サブ + volume サブ → e2e + ドキュメント** の順で進める。Step 内の **並列可能** マークは独立着手可能なペアを示す。各 Step で unit test を同時にコミットし、`dev/test-backend` をグリーンに維持する。

---

## Step 1: vast.run_meta.RunMetadata に runpod field 追加 (後方互換)

**Target**: backend
**Dependencies**: None

### Overview
`backend/src/vast/run_meta.RunMetadata` に optional フィールド (`runpod_pod_id`, `runpod_offer_snapshot`) を追加。`schema_version=1` を維持し、既存 vast run.json の読み書きに影響を与えない。

### Work Items
- [ ] `backend/src/vast/run_meta.py` の `RunMetadata` dataclass に `runpod_pod_id: str | None = None` と `runpod_offer_snapshot: dict[str, Any] | None = None` を追加 (default で後方互換)。
- [ ] 既存 `backend/tests/src/vast/test_run_meta.py` に新 field の round-trip テスト追加 (vast 既存テストはそのまま pass)。
- [ ] field 順序: `vast_*` の直後に `runpod_*` を置き、JSON 出力の可読性を確保。

### Target Files
- `backend/src/vast/run_meta.py`
- `backend/tests/src/vast/test_run_meta.py`

### Acceptance Criteria
- 既存 vast run.json (`runpod_pod_id`/`runpod_offer_snapshot` 未含) を `read_run_json` で読めば default `None` で埋まる。
- `write_run_json` → `read_run_json` の round-trip で値が保持される。
- `dev/test-backend` グリーン。

---

## Step 2: train.py の RunPod env 検出を case1/case3/case4 で追加 (並列可能 with Step 3)

**Target**: backend
**Dependencies**: Step 1

### Overview
3 つの train.py (case1/case3/case4) で `ORBIT_WARS_RUNPOD_POD_ID` / `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT` env を検出し、`RunMetadata` の対応 field を埋める。両 provider の env が同時セットされたら明示エラー (Risk 防御)。

### Work Items
- [ ] `backend/pipeline/imitation/case1/training/train.py` に env 検出ロジック追加:
  ```python
  vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
  rp_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
  if vast_id and rp_id:
      raise RuntimeError(
          "Both ORBIT_WARS_VAST_INSTANCE_ID and ORBIT_WARS_RUNPOD_POD_ID are set. "
          "Only one provider should be active per run."
      )
  ```
- [ ] `runpod_offer_snapshot` を `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT` env (JSON 文字列) から `json.loads`。空/未設定なら `None`。
- [ ] case3 / case4 の train.py にも同じパッチを適用 (該当箇所は env 解決ブロック)。
- [ ] tests:
  - `backend/tests/pipeline/imitation/case1/training/test_train_run_dir.py` に新ケース:
    - `ORBIT_WARS_RUNPOD_POD_ID="abc"` のみ → run.json に `runpod_pod_id="abc"`、`vast_instance_id=None`。
    - 両方セット → `RuntimeError`。
    - `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT='{"gpu_type_id":"X","dph_total":0.5}'` で snapshot が dict に展開される。
  - case3/case4 にも同等 mini test を追加 (既存 test 構造に合わせる)。

### Target Files
- `backend/pipeline/imitation/case1/training/train.py`
- `backend/pipeline/imitation/case3/training/train.py`
- `backend/pipeline/imitation/case4/training/train.py`
- 各 case の `tests/` 配下の `test_train_run_dir.py` (or 同等)

### Acceptance Criteria
- 既存 vast 用テスト (vast env のみ) は全て pass。
- 新規 RunPod env テストが pass。
- `dev/test-backend` グリーン。

---

## Step 3: backend/src/runpod_io/auth.py + tests (並列可能 with Step 2)

**Target**: backend
**Dependencies**: None

### Overview
`load_runpod_api_key()` を実装。`load_aws_creds()` は vast.auth から re-export して重複を避ける。

### Work Items
- [ ] `backend/src/runpod_io/__init__.py` を作成、docstring に SDK alias 規約 (`import runpod as runpod_sdk`) を明記。
- [ ] `backend/src/runpod_io/auth.py`:
  - `from vast.auth import load_aws_creds, AwsCreds, CredentialsError, DEFAULT_AWS_PROFILE, DEFAULT_AWS_REGION` で再 export。
  - `load_runpod_api_key(*, env_path: Path | None = None) -> str` 実装 (`backend/.env` → env fallback → actionable error)。
- [ ] `backend/.env.example` に `RUNPOD_API_KEY=your-runpod-api-key` 行を追加。
- [ ] `backend/pyproject.toml` の `dependencies` に `runpod>=1.7.0` 追加。
- [ ] `backend/pyproject.toml` の `[tool.hatch.build.targets.wheel] packages` (or `[tool.setuptools.packages.find]` 等) に `src/runpod_io` 追加。
- [ ] `backend/tests/src/runpod_io/__init__.py` 作成。
- [ ] `backend/tests/src/runpod_io/test_auth.py`:
  - `RUNPOD_API_KEY` 未設定で `CredentialsError` (actionable message に `runpod.io/console/user/settings` を含む)。
  - `.env` ファイルから読める (`tmp_path` fixture + dotenv 経由)。
  - 環境変数 fallback。

### Target Files
- `backend/src/runpod_io/__init__.py`
- `backend/src/runpod_io/auth.py`
- `backend/.env.example`
- `backend/pyproject.toml`
- `backend/tests/src/runpod_io/__init__.py`
- `backend/tests/src/runpod_io/test_auth.py`

### Acceptance Criteria
- `cd backend && uv sync` が `runpod` をインストール。
- `python -c "import runpod_io; import runpod"` 両方成功。
- pytest pass。

---

## Step 4: backend/src/runpod_io/run_meta.py + tests

**Target**: backend
**Dependencies**: Step 1, Step 3

### Overview
`vast.run_meta` から `RunMetadata`, `generate_run_id`, `hash_params`, `read/write/update_run_json` を re-export し、RunPod 用の helper `build_runpod_offer_snapshot()` を追加。

### Work Items
- [ ] `backend/src/runpod_io/run_meta.py`:
  ```python
  from vast.run_meta import (
      SCHEMA_VERSION, RUN_ID_PATTERN, RunMetadata, RunStatus,
      generate_run_id, hash_params,
      read_run_json, update_run_json, write_run_json,
  )
  ```
  + `build_runpod_offer_snapshot()` を実装。
- [ ] `backend/tests/src/runpod_io/test_run_meta.py`:
  - re-export が動くことを smoke test。
  - `build_runpod_offer_snapshot()` が期待通りの dict を返す。

### Target Files
- `backend/src/runpod_io/run_meta.py`
- `backend/tests/src/runpod_io/test_run_meta.py`

### Acceptance Criteria
- `from runpod_io.run_meta import RunMetadata, generate_run_id` が動く。
- pytest pass。

---

## Step 5: backend/src/runpod_io/offers.py + tests (並列可能 with Step 6)

**Target**: backend
**Dependencies**: Step 3

### Overview
RunPod 固有の 2 段階 GPU 検索 (`get_gpus()` → `get_gpu(id)`) を `Offer` dataclass にラップ。`SECURE` / `COMMUNITY` / `ALL` の cloud_type filter を実装。

### Work Items
- [ ] `backend/src/runpod_io/offers.py`:
  - `Offer` dataclass (gpu_type_id, display_name, memory_gb, secure_cloud, community_cloud, secure_price, community_price, secure_spot_price, community_spot_price, cloud_type, dph_total, data_center_id)。
  - `search_offers(sdk, *, gpu_names, cloud_type="SECURE", max_dph=2.0, min_memory_gb=16, limit=10) -> list[Offer]` を実装:
    - `sdk.get_gpus()` で id 一覧取得。
    - 各 id について `sdk.get_gpu(id)` で価格取得。
    - cloud_type に応じて `secure_price` / `community_price` のどちらを採用するか分岐。両方含む `ALL` ならそれぞれ Offer を生成 (1 GPU から最大 2 Offer)。
    - `dph_total` 昇順で `limit` 件返す。
  - `format_table()` で rich.Table 整形 (#, gpu_type_id, display_name, memory_gb, cloud_type, dph)。
  - `pick_offer()` で IntPrompt 番号入力。
- [ ] `backend/tests/src/runpod_io/test_offers.py`:
  - `sdk.get_gpus` / `sdk.get_gpu` を mock し、固定 dict で Offer 変換を検証。
  - `cloud_type=SECURE` で community-only GPU が除外される。
  - dph asc ソート確認。
  - `min_memory_gb` フィルタ効果。

### Target Files
- `backend/src/runpod_io/offers.py`
- `backend/tests/src/runpod_io/test_offers.py`

### Acceptance Criteria
- pytest pass。
- mypy / ruff pass。

---

## Step 6: backend/src/runpod_io/instance.py + onstart.sh.tmpl + tests (並列可能 with Step 5)

**Target**: backend
**Dependencies**: Step 3

### Overview
onstart テンプレ render と `create_pod` ラッパ。vast の `instance.py` を参考に同じ shell injection 対策を適用。

### Work Items
- [ ] `backend/src/runpod_io/instance.py`:
  - 9 placeholder の正規表現バリデート (vast.instance と同じ regex)。
  - `render_onstart()` (vast 同等)。
  - `build_env_dict()` (vast 同等、env 名 regex)。
  - `create_pod(sdk, *, name, gpu_type_id, cloud_type, onstart_script, env, image, container_disk_gb, network_volume_id, volume_mount_path, ports) -> str`:
    - `docker_args = f"bash -c {shlex.quote(onstart_script)}"`。
    - `sdk.create_pod(...)` 呼び出し、応答から pod id 抽出。
- [ ] `backend/src/runpod_io/onstart.sh.tmpl`:
  - vast.onstart.sh.tmpl から差分のみ:
    - 冒頭 `INSTANCE_ID="${RUNPOD_POD_ID:-unknown}"`。
    - 自殺タイムアウト保険: `( sleep 7200 && runpodctl stop pod "$INSTANCE_ID" 2>/dev/null || true ) & TIMEOUT_GUARD_PID=$!`。
    - `cleanup_destroy()` で `runpodctl stop pod "$INSTANCE_ID"` (vastai destroy の差し替え)。kill timeout guard PID。
    - `uv pip install vastai` の行は削除。
    - `ORBIT_WARS_VAST_INSTANCE_ID` を `ORBIT_WARS_RUNPOD_POD_ID` に rename。
    - `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT` を train env に追加 (JSON 文字列を渡す)。
    - bot user.email/name を `runpod-bot@orbit-wars.local` に変更。
- [ ] `backend/tests/src/runpod_io/test_instance.py`:
  - `render_onstart()` の placeholder 置換が全 9 箇所で成功。
  - shell injection 試行 (`; rm -rf /`) で `TemplateError`。
  - `build_env_dict()` の env 名 validation。
  - `create_pod()` の SDK 呼び出し引数を mock で検証。
- [ ] `backend/tests/src/runpod_io/test_onstart_template.py`:
  - `bash -n backend/src/runpod_io/onstart.sh.tmpl` でテンプレ自体が syntax error なし (subprocess)。
  - placeholder を valid 値で置換した後の bash -n も pass。

### Target Files
- `backend/src/runpod_io/instance.py`
- `backend/src/runpod_io/onstart.sh.tmpl`
- `backend/tests/src/runpod_io/test_instance.py`
- `backend/tests/src/runpod_io/test_onstart_template.py`

### Acceptance Criteria
- pytest pass。
- shell injection 試行は全て例外。
- bash -n pass。

---

## Step 7: backend/src/runpod_io/volumes.py + tests

**Target**: backend
**Dependencies**: Step 3

### Overview
RunPod の network volume CRUD は SDK が薄いため、`runpod.api.graphql.run_graphql_query` を使って GraphQL 直叩き。

### Work Items
- [ ] `backend/src/runpod_io/volumes.py`:
  - `Volume` / `VolumeOffer` dataclass。
  - `list_volumes(sdk) -> list[Volume]`: GraphQL `query MyVolumes { myself { networkVolumes { id name size dataCenterId } } }`。
  - `search_volume_offers(sdk, *, min_size_gb, data_center_id) -> list[VolumeOffer]`: 公式 docs の data center リストを返す (後で SDK 拡張があれば差し替え)。
  - `create_volume(sdk, *, name, size_gb, data_center_id) -> str`: GraphQL `mutation CreateVolume($input: NetworkVolumeInput!) { createNetworkVolume(input: $input) { id } }`。
  - `find_volume_by_name(volumes, name) -> Volume | None`。
  - `render_volume_offers()` / `pick_volume_offer()` で対話 UX。
  - `validate_volume_name(name)`: RunPod 公式制約は要確認 (Vast.ai と同じ alphanumeric+underscore で仮置き)。
- [ ] `backend/tests/src/runpod_io/test_volumes.py`:
  - `run_graphql_query` を mock、固定応答で Volume / VolumeOffer 変換を検証。
  - `find_volume_by_name` の name 一致と複数一致時の挙動。

### Target Files
- `backend/src/runpod_io/volumes.py`
- `backend/tests/src/runpod_io/test_volumes.py`

### Acceptance Criteria
- pytest pass。

---

## Step 8: backend/src/runpod_io/cost.py + tests

**Target**: backend
**Dependencies**: Step 3

### Overview
`vast.cost` を参考に、`runpod_offer_snapshot != null` の run のみを集計する `aggregate_runs()` を実装。出力は `docs/experiment/runpod_cost_report_<YYYY-MM>.md`。

### Work Items
- [ ] `backend/src/runpod_io/cost.py`:
  - `RunCost` / `CostReport` dataclass。
  - `_load_run(run_json)`: `runpod_offer_snapshot` が None なら `None` 返す (vast run スキップ)。
  - `aggregate_runs(runs_root, month=None) -> CostReport`。
  - `render_markdown(report) -> str`。
  - `parse_month()` / `iter_run_dirs()` (vast.cost と同等)。
- [ ] `backend/tests/src/runpod_io/test_cost.py`:
  - 3 個の固定 run.json (うち 1 個は vast、2 個は runpod) で、runpod 2 個のみ集計されることを検証。
  - 月単位 filter の境界条件。
  - markdown 出力の構造。

### Target Files
- `backend/src/runpod_io/cost.py`
- `backend/tests/src/runpod_io/test_cost.py`

### Acceptance Criteria
- pytest pass。

---

## Step 9: backend/src/runpod_io/cli.py (stub) + __main__.py + dev/runpod thin wrapper

**Target**: backend, dev tooling
**Dependencies**: Step 3, Step 4, Step 5, Step 6, Step 7, Step 8

### Overview
typer CLI の wiring。各サブコマンド (`train` / `pull` / `promote` / `cost-report` / `volume {list,search,create}`) を空のシグネチャで先に登録、実装は次の Step 10-13 で詰める。

### Work Items
- [ ] `backend/src/runpod_io/__main__.py`: `from runpod_io.cli import app; app()`。
- [ ] `backend/src/runpod_io/cli.py`:
  - typer.Typer() with `train`, `pull`, `promote`, `cost-report` の各サブコマンドを stub で定義。
  - `volume_app` を `app.add_typer(volume_app, name="volume")` で登録、`list`, `search`, `create` を stub 登録。
  - `CASE_DEFAULTS` 辞書を vast.cli から完コピ (中身は同じパス、placeholder 文字列も同じ)。
  - 共通 helper (`_repo_root()`, `_git()`, `_verify_commit_pushed()`, `_runs_root_for(case)`, `_case_defaults(case)`, `_build_sdk(api_key)`) を vast.cli と同等実装。
- [ ] `dev/runpod` (bash thin wrapper):
  ```bash
  #!/bin/bash
  set -euo pipefail
  cd "$(dirname "$0")/.."
  exec uv run --directory backend python -m runpod_io "$@"
  ```
- [ ] `chmod +x dev/runpod`。
- [ ] `backend/tests/src/runpod_io/test_cli.py`: `CliRunner` で `python -m runpod_io --help` が 5 サブコマンド (train/pull/promote/cost-report/volume) を表示することを smoke test。

### Target Files
- `backend/src/runpod_io/__main__.py`
- `backend/src/runpod_io/cli.py`
- `dev/runpod`
- `backend/tests/src/runpod_io/test_cli.py`

### Acceptance Criteria
- `cd backend && uv run python -m runpod_io --help` が 5 サブコマンド表示。
- `dev/runpod --help` が同じ出力。
- pytest pass。

---

## Step 10: cli.py train サブコマンド実装

**Target**: backend
**Dependencies**: Step 9

### Overview
`runpod train <sha>` で全工程を実行。

### Work Items
- [ ] `train()` 実装:
  1. `_case_defaults(case)` で stage / train_module / config_arg / preprocess_cmd / canonical_weights を解決。
  2. `_git_remote_url()` / `_git_current_branch()` / `_verify_commit_pushed()`。
  3. `load_aws_creds(profile)` / `load_runpod_api_key()`。
  4. `_build_sdk(api_key)`: `import runpod as runpod_sdk; runpod_sdk.api_key = api_key; return runpod_sdk` (module 自体を sdk として返す、SDK は module-level state)。
  5. Volume 解決: `--volume-id` / `--volume-name` 一致再利用 / `--auto-create-volume` の 3 択 (vast.cli.train 同設計)。
  6. `search_offers(sdk, gpu_names=..., cloud_type=..., max_dph=...)` → `pick_offer()`。
  7. 推定コスト確認 (`dph * 0.5h > cost_limit_usd` で `typer.confirm`)。
  8. `generate_run_id(branch, commit_sha, seed)`。
  9. `render_onstart(template_path, ...)`。
  10. `build_env_dict({"AWS_*", "RUNPOD_API_KEY", "ORBIT_WARS_RUN_ID", "ORBIT_WARS_GIT_SHA", "ORBIT_WARS_GIT_BRANCH", "ORBIT_WARS_CASE", "ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT"})`。
  11. `create_pod(sdk, name=run_id, gpu_type_id=chosen.gpu_type_id, cloud_type=chosen.cloud_type, onstart_script=onstart_cmd, env=env, image=image, container_disk_gb=disk_gb, network_volume_id=volume_id_resolved, volume_mount_path=mount_path, ports=DEFAULT_PORTS)`。
  12. 起動メッセージ表示 (`runpodctl pod logs <pod_id>` の案内 + `dev/runpod pull <run_id>` の次手順)。
- [ ] `backend/tests/src/runpod_io/test_cli.py` に `test_train_*` を追加:
  - `runpod_io.offers.search_offers` / `instance.create_pod` / `auth.*` を mock。
  - 正常 path で `create_pod` の引数を assert。
  - unpushed sha → `BadParameter`。
  - cost limit 超過 → `confirm` 経由。
  - `--cloud-type=COMMUNITY` で network volume が None になる挙動。

### Target Files
- `backend/src/runpod_io/cli.py`
- `backend/tests/src/runpod_io/test_cli.py`

### Acceptance Criteria
- pytest pass。
- mock 環境下で全エラー path が actionable メッセージ。

---

## Step 11: cli.py pull サブコマンド実装

**Target**: backend
**Dependencies**: Step 9

### Overview
vast.cli.pull の単純コピー。

### Work Items
- [ ] `pull()` 実装: `dvc pull data/output/.../runs/<run_id>` を `subprocess.run` で実行 → run.json を rich pretty print → status 警告。
- [ ] tests: `subprocess.run` を mock で正常 / 異常 path。

### Target Files
- `backend/src/runpod_io/cli.py`
- `backend/tests/src/runpod_io/test_cli.py`

### Acceptance Criteria
- pytest pass。

---

## Step 12: cli.py promote サブコマンド実装

**Target**: backend
**Dependencies**: Step 11

### Overview
vast.cli.promote の単純コピー。

### Work Items
- [ ] `promote()` 実装: `<run_dir>/best.pt` 存在確認 → `cp` to canonical → `dvc commit` → `update_run_json(status="adopted", local_eval_results=...)` → `dvc add run_dir` → `git status` 表示。
- [ ] tests: cp / dvc commit の subprocess を mock、`run.json` の status 更新を assert。

### Target Files
- `backend/src/runpod_io/cli.py`
- `backend/tests/src/runpod_io/test_cli.py`

### Acceptance Criteria
- pytest pass。

---

## Step 13: cli.py cost-report + volume {list,search,create} サブコマンド実装

**Target**: backend
**Dependencies**: Step 7, Step 8, Step 9

### Overview
残りの 4 サブコマンドを実装。

### Work Items
- [ ] `cost_report_cmd()`: `cost.aggregate_runs` → `render_markdown` → `docs/experiment/runpod_cost_report_<YYYY-MM>.md` に保存。
- [ ] `volume_app` の各サブコマンド (list/search/create) を `volumes.list_volumes` / `search_volume_offers` / `create_volume` に wire。
- [ ] tests:
  - cost-report の正常 path (固定 run.json fixtures で出力検証)。
  - volume {list, search, create} の typer 引数解釈 + SDK 呼び出し引数 assert。

### Target Files
- `backend/src/runpod_io/cli.py`
- `backend/tests/src/runpod_io/test_cli.py`

### Acceptance Criteria
- pytest pass。
- `dev/runpod cost-report --month 2026-05 --case case1` が markdown 出力。

---

## Step 14: e2e dry-run + ドキュメント

**Target**: cross-cutting
**Dependencies**: Step 10, 11, 12, 13

### Overview
本物の RunPod pod 起動はせずに、CLI の wiring と `dev/test-backend` グリーンを保つ。実環境 e2e はユーザ判断で 1 度だけ手動実行。

### Work Items
- [ ] `dev/test-backend` を実行してグリーン確認 (format / lint / type / pytest)。
- [ ] `docs/plans/runpod-basis/README.md` を作成し、運用フローを 1 ページに集約 (vast-ai-basis/README.md と同フォーマット)。
- [ ] `.claude/CLAUDE.md` の Folder Structure / Commands セクションに `dev/runpod` 追加 (vast の隣)。
- [ ] `.claude/rules/command.md` の "RunPod GPU Training" セクションを新設し、`dev/runpod train` / `dev/runpod pull` / `dev/runpod promote` / `dev/runpod cost-report` / `dev/runpod volume *` を documented。
- [ ] `backend/pipeline/imitation/case1/README.md` (ある場合) に「両基盤対応」のリファレンス追加。
- [ ] mypy / ruff / pytest が `runpod_io/` を含めて pass。

### Target Files
- `docs/plans/runpod-basis/README.md`
- `.claude/CLAUDE.md`
- `.claude/rules/command.md`
- `backend/pipeline/imitation/case1/README.md` (任意)

### Acceptance Criteria
- `dev/test-backend` グリーン。
- README が運用手順を網羅。
- CLAUDE.md と rules/command.md が `dev/runpod` を案内。

---

## Step 15: 手動 e2e (オプション、ユーザ判断)

**Target**: cross-cutting (manual)
**Dependencies**: Step 14

### Overview
本物の RunPod pod 起動を 1 度行い、onstart → dvc push → ローカル pull → promote までを手動検証。**コスト < $0.5 想定**。

### Work Items
1. `backend/.env` に `RUNPOD_API_KEY` 設定。
2. RunPod Web UI で network volume を 1 個作成 (DC: US-KS-2, size: 15GB, name: `orbit_wars_cache_runpod`)。`backend/.env` に `RUNPOD_NETWORK_VOLUME_ID=<id>` を追加 (任意 step、もしくは `dev/runpod volume create` で実装後)。
3. `feature/runpod-basis` ブランチで実装を commit & push。
4. `dev/runpod train <sha> --case case1 --cloud-type SECURE --seed 0 --label e2e-test`。
5. `runpodctl pod logs <id>` で 10-15 分監視。
6. **検証ポイント**:
   - Pod が destroy されたか (`runpodctl pod list` で消えていること)。
   - DVC remote (`s3://orbit-wars-dvc-...`) に新 hash が増えたか。
   - `dev/runpod pull <run_id>` で best.pt + metrics.json + run.json 取得可能。
   - `run.json.status="pushed"`, `runpod_pod_id` 非 None, `runpod_offer_snapshot.cloud_type="SECURE"`。
   - 推定コスト < $0.5。
7. `dev/runpod promote <run_id>` で `policy/weights.pt` 更新確認、`git status` 表示。
8. 結果を `docs/experiment/<run_id>.md` に記録、PR description にリンク。

### Acceptance Criteria
- 全手順成功。
- run.json に正しい RunPod field 群が記録される。
- 想定コスト内。

---

## Cross-Step Dependencies

```
Step 1 (vast.run_meta extend)
   │
   ├─> Step 2 (train.py × 3 case) ─────┐
   │                                    ▼
   ├─> Step 4 (runpod_io.run_meta) ─────┤
   │                                    │
Step 3 (auth.py + .env + pyproject) ────┤
   │                                    │
   ├─> Step 5 (offers.py)  ────────┐    │
   ├─> Step 6 (instance.py +       │    │
   │           onstart.sh.tmpl)    │    │
   ├─> Step 7 (volumes.py)         │    │
   └─> Step 8 (cost.py)            │    │
                                   │    │
                                   ▼    ▼
                            Step 9 (cli stub + dev/runpod)
                                   │
                  ┌────────┬───────┴───────┬────────┐
                  ▼        ▼               ▼        ▼
              Step 10   Step 11        Step 12   Step 13
              (train)   (pull)         (promote) (cost+volume)
                  │        │               │        │
                  └────────┴───────────────┴────────┘
                                   │
                                   ▼
                          Step 14 (docs + lint)
                                   │
                                   ▼
                          Step 15 (manual e2e, optional)
```

並列推奨ペア: (Step 2, Step 3)、(Step 5, Step 6, Step 7, Step 8 は Step 3 完了後に独立)、(Step 10〜13 は Step 9 完了後に並列可能)。
