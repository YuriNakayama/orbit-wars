# runpod-basis — Codebase Research

本ドキュメントは Vast.ai 基盤 (`docs/plans/vast-ai-basis/`、実装は `backend/src/vast/`) を踏襲して RunPod 基盤を作るための、既存実装の詳細分析。RunPod 基盤は Vast 基盤のミラー実装になる前提で、再利用ポイントと差分を明示する。

## Deep Codebase Analysis

### Vast.ai CLI パッケージ (`backend/src/vast/`)

- **Files analyzed**:
  - `backend/src/vast/__init__.py` (10 行) — public API のドキュメント。
  - `backend/src/vast/__main__.py` (8 行) — `python -m vast` エントリポイント。
  - `backend/src/vast/cli.py` (670 行) — typer の Typer() に `train` / `pull` / `promote` / `cost-report` / `volume {list,search,create}` を実装。case ごとの `CASE_DEFAULTS` 辞書 (case1/case3/case4) を持ち、`--case` で切替。
  - `backend/src/vast/auth.py` (116 行) — `aws configure get` をサブプロセスで叩く `load_aws_creds(profile)` と、`backend/.env` から `VAST_API_KEY` を `python-dotenv` で読む `load_vast_api_key()`。失敗時は actionable な `CredentialsError`。
  - `backend/src/vast/offers.py` (173 行) — `Offer` dataclass と `search_offers(sdk, gpu_names, max_dph, ...)`。SDK の `sdk.search_offers(query=..., type='on-demand', order='dph_total', limit=10)` をラップ。`pick_offer()` で rich Prompt 番号入力。
  - `backend/src/vast/instance.py` (181 行) — `render_onstart()` で `<COMMIT_SHA>` 等 9 placeholder を `str.replace` 置換、shell injection 対策に `_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")` で値を厳格バリデート。`build_env_dict()` は env 名を `^[A-Z_][A-Z0-9_]*$` でチェック。`create_instance(sdk, offer_id, ...)` は `sdk.create_instance(offer_id, image=..., disk=..., env=..., label=..., onstart_cmd=..., runtype="ssh_direc ssh_proxy", volume_info=...)` を呼ぶ。
  - `backend/src/vast/run_meta.py` (153 行) — `RunMetadata` dataclass (`schema_version=1`)、`generate_run_id(branch, sha, seed)` で `<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>` 形式生成、`hash_params()` (params.yaml の sha256 先頭 12 桁)、`write_run_json` / `read_run_json` / `update_run_json` (atomic rename)。
  - `backend/src/vast/cost.py` (157 行) — `aggregate_runs(runs_root, month=YYYY-MM)` で `runs/*/run.json` を全走査、`vast_offer_snapshot.dph_total × train_metrics.runtime_seconds / 3600` でコスト算出、`render_markdown()` で表化、`docs/experiment/vast_cost_report_<YYYY-MM>.md` に保存。
  - `backend/src/vast/volumes.py` (239 行) — `VolumeOffer` / `Volume` dataclass、`search_volume_offers(sdk, network=True)` で `sdk.search_network_volumes(query=...)` を呼ぶ。`create_volume(sdk, offer_id, size_gb, name, network=True)`、`list_volumes(sdk)` で所有 volume 一覧、`find_volume_by_name()` で再利用検索。`validate_volume_name()` は `^[A-Za-z0-9_]{1,64}$` 制約 (Vast.ai 固有制約)。
  - `backend/src/vast/onstart.sh.tmpl` (207 行) — bash テンプレ。`set -euo pipefail` + `trap cleanup_destroy EXIT` で「成功時のみ self-destroy、失敗時は残す」。`/persist` がマウントされていれば uv cache / DVC cache / data/mart を symlink で永続化。`uv sync --locked --no-dev --directory backend` を 3 回 retry、`uv run dvc pull data/lake/kaggle_episodes/matches.dvc` でデータ取得、`python -m <TRAIN_MODULE>` 直叩き (dvc repro は使わない理由: outs 厳密チェックが run dir override と衝突)、最後に `dvc add` + `dvc push` + `git push` (生成された `<RUN_ID>.dvc` を origin に push) → `vastai destroy instance` で自滅。

- **Current implementation**: ローカルから 1 コマンド (`dev/vast train <sha>`) で `(a) git push 確認 → (b) credentials load → (c) network volume 解決 → (d) GPU offer 検索 → (e) ユーザに番号選択させる → (f) onstart テンプレ render → (g) create_instance` を完了し、Vast 側で onstart が `clone → uv sync → dvc pull → train → dvc push → git push <RUN_ID>.dvc → self destroy` を実行。後続は `dev/vast pull <run_id>` (DVC pull) → ローカル評価 → `dev/vast promote <run_id>` (canonical weights.pt にコピー + dvc commit + git status 表示) で人間確認。

- **Key interfaces**:
  - SDK 入口: `vastai.VastAI(api_key=...)` を `_build_sdk()` 内で遅延 import (`from vastai import VastAI`)。
  - Offer 検索: `sdk.search_offers(query: str, type='on-demand', order='dph_total', limit=10) -> list[dict]`。query は `"gpu_name in [RTX_3090,RTX_4090,RTX_A6000,A100] num_gpus=1 reliability>=0.99 cuda_max_good>=12.0 dph_total<1.0 rentable=true verified=true"` のスペース区切り構文。
  - Instance 作成: `sdk.create_instance(offer_id, image=..., disk=..., env=dict, label=..., onstart_cmd=str, runtype=..., volume_info=dict|None)` 。
  - Volume: `sdk.search_network_volumes(query={...})`、`sdk.create_network_volume(id=, size=, name=)`、`sdk.show_volumes()`。

- **Patterns used**:
  - Dataclass を frozen で immutable に保つ (Offer / VolumeOffer / Volume / AwsCreds / RunMetadata)。
  - SDK 依存は `Any` 型 + 関数引数で受け取り、テスト時は mock を渡す (`def search_offers(sdk: Any, ...)`)。
  - typer の `--case` で case-specific ロジックを `CASE_DEFAULTS: dict[str, dict[str, str]]` テーブルから取得。case 追加 = 辞書追加だけで CLI は無改修。
  - `_repo_root()` は `backend/` の親 + `.git` 存在確認で resolve。`__file__.resolve().parents` 探索。
  - `subprocess.run([...], check=True, capture_output=True)` で外部コマンドを呼び、失敗は `CalledProcessError` をそのまま伝播 or actionable な例外に再 raise。
  - rich Console / Table / IntPrompt で対話 UX (terminal 上の表表示と数字入力)。

- **Coupling & side effects**:
  - `cli.py` から `auth` / `offers` / `instance` / `run_meta` / `volumes` / `cost` を全部 import している central wiring 構造。新しい backend (RunPod) を追加するなら、`cli.py` 同等の wiring と各 helper を別パッケージに切り出すのが自然。
  - `dev/vast` は `cd backend && exec uv run python -m vast "$@"` の thin wrapper のみ。`dev/runpod` も同じパターンで作れる。
  - `onstart.sh.tmpl` は **Vast.ai 固有の `VAST_CONTAINERLABEL` env や `vastai destroy instance` コマンド** に依存。RunPod 版は `RUNPOD_POD_ID` env と `runpodctl stop pod` (or `runpod.terminate_pod` SDK 呼び出し) に置換が必要。
  - DVC remote (`s3://orbit-wars-dvc-...`)、AWS profile 名 (`orbit-wars`)、`pipeline/imitation/case<N>/policy/weights.pt` への canonical path、`data/output/models/imitation/case<N>/runs/<run_id>/` の run dir 規約は **既存設計** で、RunPod 基盤も同じパスを使う (これにより `dev/vast pull` と `dev/runpod pull` で取れる成果物が完全に同じ scheme になる)。
  - `train.py` 側は `ORBIT_WARS_RUN_DIR`, `ORBIT_WARS_RUN_ID`, `ORBIT_WARS_GIT_SHA`, `ORBIT_WARS_GIT_BRANCH`, `ORBIT_WARS_VAST_INSTANCE_ID`, `ORBIT_WARS_COMMAND`, `ORBIT_WARS_CASE` の env を読んで run.json を書く設計。**`ORBIT_WARS_VAST_INSTANCE_ID` という名前は Vast 固有** なので、ここは抽象化が必要 (例: `ORBIT_WARS_VAST_INSTANCE_ID` を維持しつつ RunPod では `ORBIT_WARS_RUNPOD_POD_ID` を併用、または共通化して `ORBIT_WARS_INSTANCE_ID` + `ORBIT_WARS_PROVIDER` の 2 本にする)。

- **Test coverage** (`backend/tests/src/vast/`):
  - `test_auth.py` (94 行): `aws configure get` を `subprocess.run` mock し AwsCreds 確認、`VAST_API_KEY` 未設定で actionable error。
  - `test_cli.py` (322 行): typer の `CliRunner` で各サブコマンドの正常/異常 path、`offers.search_offers` / `instance.create_instance` / `auth.*` を `unittest.mock.patch` で stub。
  - `test_cost.py` (142 行): 固定 run.json fixture から月単位集計、markdown 出力。
  - `test_instance.py` (139 行): `render_onstart` の placeholder 置換、shell injection 試行 (`; rm -rf /`) で `TemplateError`。
  - `test_offers.py` (147 行): SDK mock で Offer 変換 + dph asc ソート。
  - `test_run_meta.py` (122 行): run_id format / params_hash 順序非依存性 / atomic write。
  - 合計 966 行のテスト。Mock パターンは流用可能。

- **Gaps identified** (RunPod 基盤を作るときに埋めるべきギャップ):
  1. **provider 抽象** — 現行 `cli.py` は Vast SDK 直叩き。runpod 基盤を後から追加するなら、共通インタフェース (search_offers, create_instance, destroy_instance) を `protocol` で切るか、各プロバイダ独立のサブパッケージ (`backend/src/runpod_io/`) にする方が clean。
  2. **`run_meta.py` の Vast 専用フィールド** — `vast_instance_id` / `vast_offer_snapshot` がある。RunPod 用に `runpod_pod_id` / `runpod_offer_snapshot` を追加するか、**汎用化して `provider`, `instance_id`, `offer_snapshot` の 3 本にリファクタする** か (既存 vast run.json との後方互換が課題)。
  3. **`onstart.sh.tmpl` の self-destroy 処理** — `vastai destroy instance "$INSTANCE_ID"` を `runpodctl stop pod $RUNPOD_POD_ID` に差し替え必要。RunPod は pod に `runpodctl` が pre-install されている (公式声明) ので、SDK install 不要になる利点がある。
  4. **`onstart.sh.tmpl` の env 名** — `VAST_CONTAINERLABEL` / `VAST_API_KEY` を読む箇所を `RUNPOD_POD_ID` / `RUNPOD_API_KEY` に変える必要。
  5. **`offers.py` の query 構文** — Vast.ai は文字列 DSL (`gpu_name in [...]`) だが、RunPod は **GPU type を id で指定** するモデル (`runpod.create_pod(gpu_type_id="NVIDIA GeForce RTX 3090")`)。RunPod では `runpod.get_gpus()` / `generate_gpu_query(gpu_id)` で list と pricing を別々に取る必要があり、フィルタロジックの再設計が必要。
  6. **コスト計算** — `vast_offer_snapshot.dph_total × runtime_seconds / 3600` は RunPod でも同じ式で計算可能 (RunPod も $/GPU/hour 表示)。`cost.py` のロジックは provider 不問で再利用可能だが、レポートタイトルやファイル名 (`vast_cost_report_*.md` → `runpod_cost_report_*.md`) を変える必要。
  7. **Volume 抽象** — Vast.ai は network/local volume を `search_network_volumes` で検索 + `create_network_volume` で作成 + 起動時 `volume_info=` で attach。RunPod の network volume は **REST/Web UI/runpodctl で別途作成** + create_pod で `network_volume_id=...` を渡す。lifecycle が独立なので、既存 `volumes.py` の API はほぼ流用可能。
  8. **Persistent volume の attach タイミング制約** — Vast は post-attach API なし、create 時のみ。RunPod も「Network volumes must be attached during Pod deployment」と同じ制約 → 設計同じで OK。

### `dev/vast` thin wrapper

- **File**: `dev/vast` (12 行)
  ```bash
  #!/bin/bash
  set -euo pipefail
  cd "$(dirname "$0")/.."
  exec uv run --directory backend python -m vast "$@"
  ```
- 設計: `dev/runpod` を同じパターン (`exec uv run --directory backend python -m runpod "$@"`) で作る。

### `train.py` (案件単位、case1/case3/case4 で別々)

- 例: `backend/pipeline/imitation/case1/training/train.py`。Vast 基盤の Step 4 で改修済み。`ORBIT_WARS_RUN_DIR` env があれば run dir に best.pt + metrics.json + run.json を書き、`vast.run_meta.RunMetadata` を使う。
- RunPod 基盤も **同じ env 規約をそのまま再利用** すれば train.py 側の改修は不要。差分は:
  - `vast_instance_id` field を持つ `RunMetadata` を、provider 不問にするか、新 field `runpod_pod_id` を追加する設計判断。
- Vast 基盤と並走する場合 (両方使う)、`train.py` から `RunMetadata` を import するパスが両基盤で共通である必要 → `vast.run_meta` をそのまま使うか、`utils/run_meta.py` のような中立位置に移すか、を Step 5 で決める。

### `pyproject.toml` (`backend/`)

- 既存依存: `vastai>=0.3.0` (line 22)、`python-dotenv`、`typer`、`rich`、`pyyaml`、`dvc[s3]`、`torch`、`numpy`、`pandas`、`polars`、`pytest` 系。
- 追加が必要: `runpod>=1.0.0` (Python SDK)。`runpodctl` は CLI バイナリでローカルには不要 (Pod 内には pre-install)。

### Plan ドキュメント既存資産

- `docs/plans/vast-ai-basis/{00-codebase-research, 01-web-research, 02-requirements, 03-architecture, 04-steps, 05-risks, 06-testing}.md` + `README.md` がフル揃い。RunPod 基盤の plan は **「Vast 基盤を踏襲し、provider 差分のみを上書く」** スタイルで書ける。

## Technical Constraints

- **Kaggle 提出は CPU 推論前提** (`backend/pipeline/*/policy/agent.py`)。RunPod 基盤も「学習 only、推論はローカル/Kaggle CPU」で OK。
- **DVC remote は S3 (`orbit-wars-dvc-...`)、profile `orbit-wars`** (Vast 基盤と完全共有)。RunPod 側でも `AWS_ACCESS_KEY_ID/SECRET` を env で渡して boto3 default chain で読ませる。
- **dvc cache は `/Users/user/project/orbit-wars/.dvc/cache` (worktree 共有)**。pull の同時実行は控える運用ルールあり (`.claude/rules/command.md` 参照)。
- **`backend/.env` は git ignore 済み**、新規 `RUNPOD_API_KEY` の追加先として安全。
- **commit-sha は origin に push 済みでなければならない** (Vast 基盤の制約と同じ。RunPod でも clone するため)。
- **ファイル数とテスト規模** — Vast 基盤は src 1.7K 行 / tests 0.97K 行で、CI 上の pytest 時間に余裕がある (`dev/test-backend` 数十秒)。RunPod 基盤を同規模で追加してもコスト無視可能。

## Key Findings Summary

- Vast 基盤は機能ごとに 1 ファイル (`auth.py` / `offers.py` / `instance.py` / `volumes.py` / `cost.py` / `run_meta.py` / `cli.py` / `onstart.sh.tmpl`) に分かれた **きれいな mirror 可能構造**。RunPod 版 `backend/src/runpod_io/` を同じファイル構成で作るのが最小差分・最大流用。
- `train.py` 側 (case1/case3/case4) は既存の env プロトコル (`ORBIT_WARS_RUN_DIR` 等) を **そのまま再利用** できる。`vast_instance_id` フィールドは `RunMetadata` にあるが、RunPod 用の新フィールドを追加するか、汎用化するかは **Step 5 (Architecture)** で決定。
- `dev/vast` の thin wrapper パターンを `dev/runpod` でも採用 — 1 行の `exec uv run --directory backend python -m runpod "$@"` で済む。
- 共通化(=`utils/cloud_run/` 等への抽象昇格)するか、独立 mirror (=`backend/src/runpod_io/`) にするかは設計判断。**MVP は独立 mirror、共通化は Phase 2** が学習リスク低い (vast 既存実装を壊さない)。
- DVC / S3 / canonical weights / run.json schema / runs/<run_id>/best.pt 規約は **provider 不問で同じパス** を使う。これにより `dev/vast pull <run_id>` と `dev/runpod pull <run_id>` がほぼ同じ振る舞い (run_id format さえ被らなければ並列に存在可能)。
- RunPod 固有の onstart 機構は **`docker_args` か image 内 `start.sh`** の 2 択で、`docker_args` は exit すると pod が再起動する罠あり → 末尾に `sleep infinity` 不要、自殺 (`runpodctl stop pod $RUNPOD_POD_ID`) で正常終了する設計が clean。
