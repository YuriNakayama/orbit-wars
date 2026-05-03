# runpod-basis — Requirements Definition

## Background and Purpose

`docs/plans/vast-ai-basis/` で構築した Vast.ai 基盤は、PR commit → GPU 学習 → 結果評価のループを 30 分単位に短縮した。一方で Vast.ai は host quality が分散しており (community marketplace)、reliability=0.99 でも切断や availability の偏りが運用上のばらつきを生んでいる。RunPod は Secure Cloud + per-minute billing + network volume が安定して使えるため、「reliability 重視で長め (例: H100 を A6000 の代わりに) を回したい」「Community との価格メリットを使い分けたい」という選択肢を持たせたい。

本 feature は **Vast.ai 基盤を完全 mirror した RunPod 基盤を `bot/src/runpod_io/` として独立実装** し、開発者が `dev/vast train` と `dev/runpod train` を価格・availability・reliability に応じて使い分けられるようにする。両基盤は coexist し、どちらも同じ DVC remote (S3 `orbit-wars-dvc-...`) と同じ run dir scheme (`data/output/models/imitation/case<N>/runs/<run_id>/`) を共有する。

副次目的:
- (a) `train.py` 側 (`bot/pipeline/imitation/case{1,3,4}/training/train.py`) を **無改修** とし、provider 中立な env プロトコル (`ORBIT_WARS_RUN_DIR` 等) をそのまま流用する。
- (b) `run.json` schema は v1 を維持し、optional field `runpod_pod_id` / `runpod_offer_snapshot` を追加 (vast 既存 run.json は影響なし、後方互換確保)。
- (c) 1 run ごとの cost を月次集計し、provider 別に `docs/experiment/runpod_cost_report_<YYYY-MM>.md` として記録 (vast 側と分離)。

## User Stories

- As a **developer**, I want to run `dev/runpod train <commit-sha> [--case case1] [--cloud-type SECURE]` from my laptop and have GPU training start on RunPod, so that I can fall back to RunPod when Vast.ai availability is poor or I need Secure Cloud reliability.
- As a **developer**, I want the CLI to call `runpod.get_gpus()` + `runpod.get_gpu(id)` and show me the cheapest 10 GPU candidates filtered by `cloud_type` / `min_memory_in_gb` / `max_price`, so I can pick by price and stability.
- As a **developer**, I want to choose `cloud_type=SECURE / COMMUNITY / ALL` per invocation, so I can balance reliability vs cost.
- As a **developer**, I want each run's outputs (best.pt + metrics.json + run.json) saved to the same `data/output/models/imitation/case<N>/runs/<run_id>/` scheme used by the Vast basis, so artifacts produced by either provider are interchangeable in `dev/vast pull` / `dev/runpod pull` / `dev/runpod promote`.
- As a **developer**, I want `dev/runpod pull <run_id>` and `dev/runpod promote <run_id>` to work the same way as the Vast equivalents, so muscle memory transfers.
- As a **researcher**, I want `run.json` to additionally record `runpod_pod_id` and `runpod_offer_snapshot` (cloud_type, gpu_type_id, secureCloud, communityCloud, dph_total) when launched via RunPod, so any past run is reproducible/auditable regardless of provider.
- As a **cost-conscious developer**, I want a soft warning when the estimated cost (`dph_total × estimated_runtime_minutes / 60`) exceeds **$1.5 USD** per run, so runaway training doesn't surprise me. (Vast の $1.0 より高めに設定 — RunPod は同 GPU で 20-30% 高いため。)
- As an **operator**, I want a monthly cost report aggregator that scans `runs/*/run.json` and produces a markdown summary for **RunPod runs only**, separately from the Vast cost report, so I can compare per-provider monthly burn.
- As a **developer**, I want the RunPod pod to **self-destroy via `runpodctl stop pod $RUNPOD_POD_ID`** at the end of onstart on success, with a tail timeout fallback (`sleep 2h; runpodctl stop pod ...` background) for safety, so idle pods don't accrue cost even if the bash trap fails.
- As a **developer**, I want the basis to coexist with the Vast basis: both `dev/vast` and `dev/runpod` are first-class CLIs, neither is deprecated, README/CLAUDE.md describes both with guidance on when to pick which.
- As a **developer**, I want `dev/runpod volume {list,search,create}` subcommands (mirroring `dev/vast volume`) to manage RunPod network volumes, so persistent uv/DVC cache is available across runs.

## Functional Requirements

### F1. CLI: `dev/runpod train <commit-sha> [--case case1] [--cloud-type SECURE] [--gpu-name ...] [--seed N] [--label TEXT]`

1. **F1.1** `<commit-sha>` 必須引数。ローカル git で存在 + origin push 済みを検証 (vast 同等)。失敗で fail-fast。
2. **F1.2** `--case` (デフォルト `case1`) で `case1` / `case3` / `case4` を切替。`CASE_DEFAULTS` テーブルは vast.cli の構造を完コピ (`stage`, `train_module`, `config_arg`, `preprocess_cmd`, `canonical_weights`)。
3. **F1.3** `--cloud-type` で `SECURE` / `COMMUNITY` / `ALL` を選択。デフォルトは `SECURE` (Secure Cloud + RTX 3090/4090/A6000 を優先)。
4. **F1.4** GPU フィルタ: `--gpu-names` で複数指定可、デフォルトは `["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A100 80GB PCIe"]`。`runpod.get_gpus()` で id 一覧取得 → 各 id について `runpod.get_gpu(id)` を呼んで価格情報取得 → `cloud_type` と一致するものから `(securePrice or communityPrice) <= max_dph` で絞り込み → dph 昇順 top 10。
5. **F1.5** rich Table で 10 件表示 (#, gpu_id, displayName, memory_gb, secureCloud, communityCloud, dph, region)。stdin で番号入力。
6. **F1.6** 推定コスト (`dph × 0.5h`) を表示。`--cost-limit` (デフォルト **$1.5**) を超えたら `typer.confirm` で再確認。
7. **F1.7** `run_id = generate_run_id(branch, commit_sha, seed)` (vast.run_meta の関数を share / 同等関数を runpod 側に置く)。
8. **F1.8** Network volume 解決:
   - `--volume-id` 明示があればそれを使用。
   - なければ `--volume-name` (デフォルト `orbit_wars`) で既存 volume を `list_volumes(sdk)` 検索 → 名前一致なら再利用。
   - `--auto-create-volume` フラグがあり、一致 volume 不在なら `search_volume_offers` → ユーザに choice → `create_volume` で新規作成。
   - どれでもなければ volume なし (uv cache / DVC cache 永続化なし)。
9. **F1.9** onstart スクリプト構築:
   - `bot/src/runpod_io/onstart.sh.tmpl` を読み、`<COMMIT_SHA>`, `<RUN_ID>`, `<STAGE>`, `<BRANCH>`, `<REPO_URL>`, `<CASE>`, `<TRAIN_MODULE>`, `<CONFIG_ARG>`, `<PREPROCESS_CMD>` の 9 placeholder を sanitize 後 `str.replace`。
   - vast 同等の正規表現バリデート (`_VALID_VALUE = ^[A-Za-z0-9._\-/:]+$`、`_VALID_CONFIG_ARG`、`_VALID_PREPROCESS_CMD`) で shell injection 対策。
10. **F1.10** env 注入: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `RUNPOD_API_KEY`, `ORBIT_WARS_RUN_ID`, `ORBIT_WARS_GIT_SHA`, `ORBIT_WARS_GIT_BRANCH`, `ORBIT_WARS_CASE`, **`ORBIT_WARS_RUNPOD_POD_ID`** (空文字、後で onstart 内で `RUNPOD_POD_ID` env を assignment)。
11. **F1.11** `runpod.create_pod(...)` 呼び出し:
    ```python
    pod = sdk.create_pod(
        name=run_id,
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=chosen.id,
        cloud_type=cloud_type,           # SECURE / COMMUNITY / ALL
        gpu_count=1,
        container_disk_in_gb=40,
        volume_in_gb=0,                   # network volume 使うので一時 disk 0
        network_volume_id=volume_id_resolved,
        volume_mount_path="/persist",     # vast と同じ mount path
        docker_args=f"bash -c {shlex.quote(onstart_script)}",
        env={...},                        # F1.10
        support_public_ip=True,
        start_ssh=True,
        ports="22/tcp,8888/http",
    )
    ```
12. **F1.12** 起動後、pod_id を表示。`runpodctl pod logs <pod_id>` で監視するモニタコマンドをユーザに案内。`dev/runpod pull <run_id>` の次手順も併記。
13. **F1.13** dirty working tree (uncommitted changes) は警告のみ (vast 同等、commit-sha は固定なので学習自体は影響なし)。

### F2. Onstart script: `bot/src/runpod_io/onstart.sh.tmpl`

vast の onstart.sh.tmpl を踏襲しつつ、self-destroy のみ RunPod 流。

1. **F2.1** `set -euo pipefail` + `exec > >(tee -a /var/log/onstart.log) 2>&1` で堅牢化 + ログ取得。
2. **F2.2** `INSTANCE_ID="${RUNPOD_POD_ID:-unknown}"` で pod 自身の id を取得 (Vast の `VAST_CONTAINERLABEL` 相当)。
3. **F2.3** **自殺タイムアウト保険**: スクリプト冒頭で `( sleep 7200 && runpodctl stop pod "$INSTANCE_ID" ) &` の background job を仕掛ける (2h hard timeout)。
4. **F2.4** EXIT trap (`cleanup_destroy`):
   - 成功 (`exit_code=0`) → `runpodctl stop pod "$INSTANCE_ID"` で正常自殺。
   - 失敗 (`exit_code != 0`) → 残してログ案内 (vast の運用ポリシーと同じ)。
   - `runpodctl` は image に pre-install されているため SDK 経由 install 不要 (vast.onstart の `uv pip install vastai` が不要になる)。
5. **F2.5** `env >> /etc/environment` で SSH デバッグ時に env 可視化。
6. **F2.6** `/persist` がマウントされていれば uv cache / DVC cache / data/mart を symlink 永続化 (vast.onstart と同一処理)。
7. **F2.7** `git clone <REPO_URL> orbit-wars` (private repo の場合 `GIT_PAT` env 経由で認証、vast と同設計)。`git checkout <COMMIT_SHA>`。
8. **F2.8** `curl -LsSf https://astral.sh/uv/install.sh | sh` で uv install (image に未含まれている前提)。
9. **F2.9** `uv sync --locked --no-dev --directory bot` を 3 回 retry (大型 wheel ダウンロード対策)。
10. **F2.10** `uv run --project bot dvc pull data/lake/kaggle_episodes/matches.dvc` + `uv run --project bot dvc pull` で deps 取得。
11. **F2.11** `mkdir -p data/output/models/imitation/<CASE>/runs/<RUN_ID>` で run dir 作成。
12. **F2.12** Preprocess: parquet が `data/mart/imitation/<CASE>/*.parquet` に存在するなら skip、なければ `<PREPROCESS_CMD>` を実行 (vast.onstart 同設計)。
13. **F2.13** Train: dvc repro は使わず `python -m <TRAIN_MODULE> <CONFIG_ARG>` を直叩き (vast 同設計)。env で `ORBIT_WARS_RUN_DIR`, `ORBIT_WARS_RUN_ID`, `ORBIT_WARS_GIT_SHA`, `ORBIT_WARS_GIT_BRANCH`, **`ORBIT_WARS_RUNPOD_POD_ID`** (Vast 側の `ORBIT_WARS_VAST_INSTANCE_ID` の RunPod 版)、`ORBIT_WARS_COMMAND` を渡す。
14. **F2.14** `dvc add data/output/.../runs/<RUN_ID>` + `dvc push` で S3 へ。
15. **F2.15** 生成された `<RUN_ID>.dvc` を `git push origin <BRANCH>` でリポジトリに反映 (3 回 rebase retry、失敗してもアーティファクトは S3 にあるので fatal にしない)。vast.onstart 同設計。
16. **F2.16** ログは各ステップで `echo "[onstart] step=... case=..."` を出力。

### F3. Run metadata: `run.json` schema 拡張 (v1 互換)

vast.run_meta.RunMetadata に optional フィールドを追加。`schema_version` は `1` を維持。

1. **F3.1** 新規 fields (Optional, デフォルト `None`):
   - `runpod_pod_id: int | None = None`
   - `runpod_offer_snapshot: dict[str, Any] | None = None` (例: `{"gpu_type_id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24, "cloud_type": "SECURE", "secureCloud": true, "communityCloud": false, "dph_total": 0.43, "data_center_id": "US-KS-2"}`)
2. **F3.2** Vast 側の既存フィールド (`vast_instance_id`, `vast_offer_snapshot`) は **そのまま維持**、両者は排他的に埋まる (片方の provider のみ非 None)。
3. **F3.3** `train.py` 側の改修は不要: `RunMetadata` のキーワード引数経由で provider に応じて埋める (env 検出ロジックは F4 で扱う)。
4. **F3.4** `cost.py` (RunPod 版) は `runpod_offer_snapshot.dph_total × runtime_seconds / 3600` で算出。`vast.cost` と関数構造は同じ。

### F4. `train.py` の RunPod env 対応

`bot/pipeline/imitation/case{1,3,4}/training/train.py` は vast 基盤改修済み。RunPod 用の追加対応:

1. **F4.1** env 検出ロジック追加: `os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")` がセットされていれば、`RunMetadata.runpod_pod_id` を埋める。`ORBIT_WARS_VAST_INSTANCE_ID` がセットされていれば従来通り `vast_instance_id` を埋める。
2. **F4.2** **両方 set** されていることはあり得ない (provider は run.json 1 つに 1 個しか出ない)。assertion として「両方 set はエラー」を入れる。
3. **F4.3** GPU 名の取得: `torch.cuda.get_device_name(0)` で取得し `gpu_name` フィールドへ (provider 不問)。
4. **F4.4** **既存 vast 用テスト (`test_train_run_dir.py`) を壊さない**。RunPod env を追加で検証するテストを追加 (テスト ID は次 Step で具体化)。

### F5. CLI: `dev/runpod pull <run_id> [--case case1]`

vast.cli.pull と同設計。`<runs_root>/<run_id>.dvc` の存在確認 → `dvc pull` → `run.json` を rich JSON 表示 → `status != "pushed"` で警告。

### F6. CLI: `dev/runpod promote <run_id> [--case case1] [--eval-results PATH]`

vast.cli.promote と同設計。`run_dir/best.pt` を `<canonical>` (`pipeline/imitation/case<N>/policy/weights.pt`) にコピー → `dvc commit` → `run.json status=adopted` 更新 → `dvc add run_dir` → `git status` 表示。`--eval-results` は JSON ファイルから `local_eval_results` フィールドへマージ。

### F7. CLI: `dev/runpod cost-report [--month YYYY-MM] [--case case1]`

`bot/src/runpod_io/cost.py` の `aggregate_runs()` で `runs/*/run.json` を全走査 → `runpod_offer_snapshot.dph_total != null` のものだけを集計 (vast の run はスキップ) → markdown を `docs/experiment/runpod_cost_report_<YYYY-MM>.md` に保存。

### F8. CLI: `dev/runpod volume {list,search,create}`

vast.cli.volume_app と同等。`runpod.api` の network volume 操作は SDK が薄いため、`runpod.api.graphql.run_graphql_query()` で生 GraphQL を叩くか、REST 直叩きで補完。

1. **F8.1** `dev/runpod volume list`: 所有 volume 一覧表示 (id / name / size_gb / data_center / $/GB/月)。
2. **F8.2** `dev/runpod volume search [--min-size 15] [--data-center-id US-KS-2]`: 利用可能なデータセンターと容量帯を表示。
3. **F8.3** `dev/runpod volume create <name> [--size 15] [--data-center-id US-KS-2]`: 新規 volume 作成、id 表示。
4. **F8.4** Volume name 制約: RunPod 側に明示制約があれば validate (Vast の alphanumeric+underscore は仕様、RunPod は確認次第バリデート追加)。

### F9. Configuration: 環境変数とローカル設定

1. **F9.1** `bot/.env` に `RUNPOD_API_KEY=<your-key>` を追加 (vast の `VAST_API_KEY` と同じ枠組み)。`bot/.env.example` にプレースホルダ行を追加。
2. **F9.2** `bot/src/runpod_io/auth.py` の `load_runpod_api_key()` は vast.auth.load_vast_api_key と同パターン (dotenv → env fallback → actionable error)。
3. **F9.3** AWS credentials は `vast.auth.load_aws_creds()` と同じ関数を流用 (profile=`orbit-wars`)。**vast/auth.py から再利用** または **runpod/auth.py で同じ関数を独立実装**。実装方式は Architecture (Step 5) で決定。
4. **F9.4** `pyproject.toml` に `runpod>=1.7.0` を依存追加。`vastai>=0.3.0` はそのまま維持 (両基盤共存)。

### F10. パッケージ命名と SDK との衝突回避

ユーザ確定: 内部パッケージ名は `runpod_io`、CLI 名は `dev/runpod`。

1. **F10.1** 内部パッケージは `bot/src/runpod_io/` に配置。SDK との衝突は **パッケージ名分離で物理的に解決**。
2. **F10.2** SDK は各モジュールで `import runpod as runpod_sdk` の alias で読む (規約)。`runpod_io` パッケージは `runpod` SDK と別名なので Python の import システムが両者を独立に解決可能。
3. **F10.3** `bot/pyproject.toml` の `[tool.hatch.build.targets.wheel]` (or 同等) に `src/runpod_io` を追加。`[tool.mypy]` / `[tool.ruff]` の対象にも追加。
4. **F10.4** thin wrapper `dev/runpod` の中身は `exec uv run --directory bot python -m runpod_io "$@"`。CLI 名は要件通り `runpod`、Python module 名は `runpod_io`。

## Non-Functional Requirements

### NFR-1. 性能 / レスポンス

- `dev/runpod train` 起動 → `create_pod` 完了までの体感 < 30 秒 (`get_gpus` ~5s + `get_gpu` ループ 5-10s + `create_pod` 10s)。
- onstart 終了までの実時間 < 30 分 (vast 同等、image / network 速度に依存)。
- 1 run の典型コスト < 0.5 USD (Secure RTX 3090 @ $0.43/h × 0.5h = $0.21、Community @ $0.22/h × 0.5h = $0.11、A100 で $0.6/h × 0.5h = $0.30)。

### NFR-2. セキュリティ

- AWS keys / `RUNPOD_API_KEY` は env 経由で pod にのみ渡す。
- IAM 権限は既存 `orbit-wars-dev-dvc-user` を再利用 (`s3:DeleteObject` なし)。
- `runpodctl` の自殺は **pod-scoped API key** で実行されるため、漏洩しても他 pod を destroy できない (RunPod 公式仕様)。
- onstart は `set -x` を使わない (env が stdout に流れる)。
- `run.json` には credentials を一切含めない。`runpod_offer_snapshot` は public 情報のみ (gpu_type_id, dph 等)。

### NFR-3. 可用性 / 障害耐性

- pod 作成失敗 (capacity なし、network volume DC 不一致等) は典型エラーを actionable メッセージに翻訳 (例: "no SECURE pods available with RTX 3090 in US-KS-2 — try `--cloud-type=COMMUNITY` or change network volume region")。
- onstart 中の `git clone` / `uv sync` / `dvc pull` 失敗は `set -e` で即時中断 → trap が **pod を残す** → ローカルから `runpodctl pod logs` で原因確認 + `runpodctl pod ssh` で sshin 可能。
- `dvc push` 失敗時は **destroy しない** (vast と同じ運用ポリシー)。
- 2h タイムアウト保険 (F2.3) で trap が壊れても pod は止まる。

### NFR-4. 拡張性

- `--case` で imitation/case{1,3,4} を切替、`CASE_DEFAULTS` 辞書追加だけで拡張可。
- Spot/interruptible は **本 feature では非対応**。将来必要になったら `--bid` フラグ追加で対応可 (RunPod の `podRentInterruptable` mutation は SDK 経由で部分的に呼べる)。
- 将来 `dev/cloud train --provider=vast|runpod` の上位 CLI を作る場合、両基盤の public API (`search_offers` / `create_instance` / `pull` / `promote` / `cost-report`) が一致しているため少ない変更で抽象化可能。

### NFR-5. 保守性

- vast 基盤と **同じファイル構成** (auth, offers, instance, run_meta, cost, cli, onstart.sh.tmpl, volumes) で並行配置。コードレビュー時に diff で機能対応関係が把握できる。
- テスト構造も同じ (test_auth, test_offers, test_instance, test_run_meta, test_cost, test_cli)。
- mypy + ruff 通過、`dev/test-bot` グリーン維持。

## Out of Scope

- **Spot / Interruptible pod**: on-demand 一択 (短時間学習なので中断耐性不要)。将来別 feature。
- **`dev/cloud train --provider=...` の上位 CLI**: 両基盤共存が前提、抽象化は Phase 2。
- **マルチ GPU 学習 (DDP)**: DeepSets MLP は単 GPU で十分。
- **Hyperparameter sweep automation**: 1 commit = 1 run 原則、本 feature の対象外。
- **Vast.ai 基盤の廃止**: 並走するため改修なし。
- **GPU 推論 (Kaggle submit 改修)**: Kaggle Sandbox は CPU 環境前提。
- **既存 weights_iter*.pt の DVC 移行**: 既存 vast 基盤と同じく段階移行。
- **REST API 直叩きでの全機能実装**: 公式 SDK + GraphQL helper で必要十分。生 REST は volume API でのみ補助的に使用。
- **provider 抽象 base class (`Cloud`/`ResourceHandle` 風)**: 完全 mirror 方針なので不要。

## Glossary

| Term | Description |
|------|-------------|
| `pod_id` | RunPod 上の pod を一意特定する文字列 (or int)。pod 内では env `RUNPOD_POD_ID` で参照可能 |
| `gpu_type_id` | RunPod の GPU 種類識別子 (例: `"NVIDIA GeForce RTX 3090"`)。`runpod.get_gpus()` から取得 |
| `cloud_type` | `SECURE` (T3/T4 DC) / `COMMUNITY` (P2P marketplace) / `ALL` の 3 値 |
| Network volume | RunPod の永続化ストレージ。Pod 作成時のみ attach 可能、Secure Cloud 専用 |
| Secure Cloud | RunPod の高信頼 DC pod。network volume が attach 可能 |
| Community Cloud | RunPod の P2P pod。安価だが network volume 不可、reliability 中 |
| `RUNPOD_API_KEY` | RunPod のアカウント API key。https://runpod.io/console/user/settings で発行 |
| `runpodctl` | RunPod CLI バイナリ。Pod 内には pre-install + pod-scoped key で `stop pod $RUNPOD_POD_ID` 可能 |
| `runpod_offer_snapshot` | run.json に保存する pod 起動時の offer メタ情報 (gpu_type_id, cloud_type, dph 等) |
| canonical weights | `bot/pipeline/imitation/case<N>/policy/weights.pt` (両基盤共通) |
| candidate weights | `<run_dir>/best.pt` (両基盤共通) |
| run dir | `data/output/models/imitation/case<N>/runs/<run_id>/` (両基盤共通の scheme) |
