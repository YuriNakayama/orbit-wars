# kaggle-kernel-basis — Requirements Definition

## Background and Purpose

`docs/plans/vast-ai-basis/` と `docs/plans/runpod-basis/` で構築した 2 つの GPU 学習基盤 (Vast.ai / RunPod) は、価格 / 安定性のトレードオフを開発者に選ばせる構造になっている。一方どちらも **従量課金** であり、imitation case1 の試行錯誤期のように 1 case を何十回も回すフェーズではコストが嵩む。Kaggle Notebooks (Kaggle Kernel) は **週 30h まで無料 GPU** が使えるため、**コスト 0** で小規模学習を回す第三の基盤として価値が高い。

本 feature は **Vast / RunPod 基盤を完全 mirror した Kaggle Kernel 基盤を `bot/src/kaggle_kernel/` として独立実装** し、開発者が `dev/vast train` / `dev/runpod train` / `dev/kaggle-kernel train` を価格・availability・reliability・無料枠の有無に応じて使い分けられるようにする。三基盤は coexist し、すべて同じ DVC remote と同じ run dir scheme (`data/output/models/imitation/case<N>/runs/<run_id>/`) を共有する。

副次目的:
- (a) `train.py` 側 (`bot/pipeline/imitation/case{1,3,4,...}/training/train.py`) の provider 検出ブロックに第三 env (`ORBIT_WARS_KAGGLE_KERNEL_SLUG`) を追加する以外は無改修。三 provider 同時セットは明示エラー。
- (b) `RunMetadata` schema は v1 を維持し、optional field `kaggle_kernel_meta` を追加 (既存 vast/runpod run.json は影響なし、後方互換確保)。
- (c) コスト集計は **金額 0** だが、月次 free GPU hour 使用量を `docs/experiment/kaggle_kernel_cost_report_<YYYY-MM>.md` として記録する。

## User Stories

- As a **developer**, I want to run `dev/kaggle-kernel train <commit-sha> [--case case1] [--accelerator gpu-t4x2]` from my laptop and have GPU training start on Kaggle Notebooks, so that I can iterate on imitation case freely without GPU cost.
- As a **developer**, I want the first invocation to auto-upload `bot/` as a Kaggle Dataset (`<user>/orbit-wars-bot`), and subsequent invocations to do `dataset_create_version` only when the commit SHA differs from the most recent uploaded version, so that dataset version churn is minimized.
- As a **developer**, I want a generated notebook to be pushed with `kaggle kernels push`, configured to `Add Data` from the bot dataset, install via `pip install -e /kaggle/input/orbit-wars-bot/`, and run `python -m pipeline.imitation.<case>.training.train ...` with appropriate env vars, so that the same training script that runs on Vast / RunPod also runs on Kaggle.
- As a **developer**, I want `dev/kaggle-kernel pull <run_id>` to fetch `/kaggle/working/runs/<run_id>/` via `kaggle kernels output`, place artifacts into `data/output/models/imitation/case<N>/runs/<run_id>/`, then `dvc add` + `dvc push`, so the artifact flow matches Vast / RunPod.
- As a **developer**, I want `dev/kaggle-kernel promote <run_id>` to be a thin shim over the shared `promote_to_canonical()` already used by `dev/runpod promote`, so muscle memory transfers.
- As a **researcher**, I want `run.json` to additionally record `kaggle_kernel_meta` (kernel_slug, kernel_version, dataset_slug, dataset_version, accelerator, runtime_seconds, internet_enabled) when launched via Kaggle Kernel, so any past run is reproducible/auditable regardless of provider.
- As a **cost-conscious developer**, I want a soft warning when the **estimated free GPU hours remaining** drops below 5h before starting a new run, so I don't get stranded mid-week.
- As an **operator**, I want a monthly free-hour usage report aggregator (`dev/kaggle-kernel cost-report`) that scans `runs/*/run.json` and produces a markdown summary for **Kaggle runs only**, separately from the Vast / RunPod cost reports.
- As a **developer**, I want `dev/kaggle-kernel ps` to list active kernels (queued + running), so I don't accidentally exceed the ~5 kernel concurrent limit.
- As a **developer**, I want `dev/kaggle-kernel logs <run_id>` to fetch stdout from completed kernels (Kaggle does not provide live streaming via API; this is post-mortem only), so I can debug after-the-fact.
- As a **developer**, I want the basis to coexist with the Vast / RunPod bases: all three CLIs are first-class, none deprecated, README/CLAUDE.md describes when to pick which.

## Functional Requirements

### F1. CLI: `dev/kaggle-kernel train <commit-sha> [--case case1] [--accelerator gpu-t4x2] [--seed N] [--label TEXT] [--no-internet] [--watch] [--dataset-bump-only] [--max-hours 8.5]`

1. **F1.1** `<commit-sha>` 必須引数。ローカル git で存在 + origin push 済みを検証 (vast/runpod 同等)。失敗で fail-fast。
2. **F1.2** `--case` (デフォルト `case1`) で `case1` / `case3` / `case4` / `case8` 等を切替。`CASE_DEFAULTS` テーブルは `runpod_io.config.cases` を再利用 (要 import 経路の整理)。
3. **F1.3** `--accelerator` で `gpu-t4x2` / `gpu-p100` / `cpu` を選択。デフォルトは `gpu-t4x2`。
4. **F1.4** `run_id = generate_run_id(branch, commit_sha, seed)` (vast.run_meta の関数を share)。
5. **F1.5** Dataset 解決:
   - `--dataset-bump-only` 明示時: 既存 dataset の最新 version を流用 (新 version 作成しない)。
   - 未指定: 最新 dataset version の commit SHA が一致しないなら `dataset_create_version` で新 version を push。
   - dataset 自体が未作成なら `dataset_create_new` で初回作成 (例外的、`dev/kaggle-kernel dataset push` の経路推奨)。
6. **F1.6** Notebook 構築 (`bot/src/kaggle_kernel/kernel/template.py`):
   - cell A: env 注入 (`os.environ.update({...})`)
   - cell B: `!pip install /kaggle/input/orbit-wars-bot/wheels/*.whl` (Rust wheel 先)
   - cell C: `!pip install -e /kaggle/input/orbit-wars-bot/`
   - cell D: `subprocess.run(["python", "-m", f"pipeline.imitation.{case}.training.train", ...], check=True)`
   - cell E: artifact を `/kaggle/working/runs/<run_id>/` にコピー、`run.json` 整形 (`kaggle_kernel_meta` を埋める)
7. **F1.7** env 注入:
   - `ORBIT_WARS_RUN_ID`, `ORBIT_WARS_GIT_SHA`, `ORBIT_WARS_GIT_BRANCH`, `ORBIT_WARS_CASE`
   - `ORBIT_WARS_KAGGLE_KERNEL_SLUG`, `ORBIT_WARS_KAGGLE_KERNEL_VERSION` (空文字、kernel push 後に上書き不可なので initial value のみ)
   - `ORBIT_WARS_KAGGLE_ACCELERATOR`
   - `ORBIT_WARS_KAGGLE_KERNEL_META` (JSON 文字列、train.py が `RunMetadata.kaggle_kernel_meta` に展開)
   - `ORBIT_WARS_RUN_DIR=/kaggle/working/runs/<run_id>`
8. **F1.8** `kernel-metadata.json` 構築:
   ```json
   {
     "id": "<user>/orbit-wars-<case>-<run_id_slug>",
     "title": "orbit-wars <case> <run_id_slug>",
     "code_file": "main.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": "true",
     "enable_gpu": "true",
     "enable_internet": "<from --no-internet flag>",
     "dataset_sources": ["<user>/orbit-wars-bot"],
     "competition_sources": [],
     "kernel_sources": []
   }
   ```
9. **F1.9** `KaggleApi().kernels_push_cli(folder)` を呼び、結果から kernel slug / version を取得。
10. **F1.10** `--watch` 指定時: 30-60s 間隔で `kernels_status` を polling、`complete` / `error` / `cancel_acknowledged` で離脱。完了で desktop 通知 (`runpod_io.notify` を共有 import)。
11. **F1.11** dirty working tree (uncommitted changes) は警告のみ (commit-sha 固定なので学習自体は影響なし)。
12. **F1.12** safety check: `KaggleApi().kernels_list(user=...)` で active 中の kernel 数を fetch、4 件以上なら typer.confirm。

### F2. Notebook template: `bot/src/kaggle_kernel/kernel/template.py`

Vast / RunPod の `onstart.sh.tmpl` に相当。bash でなく Python の jupyter notebook (.ipynb) を組み立てる。

1. **F2.1** placeholder 置換は `<COMMIT_SHA>`, `<RUN_ID>`, `<CASE>`, `<TRAIN_MODULE>`, `<CONFIG_ARG>`, `<DATASET_SLUG>`, `<DATASET_VERSION>`, `<ACCELERATOR>` の 8 種。`str.replace` で展開。
2. **F2.2** shell injection 対策: env 値は `_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:=]+$")` でバリデート。
3. **F2.3** cell A は `os.environ.update({...})` で全 env を一括設定。
4. **F2.4** cell B は `!pip install -q /kaggle/input/orbit-wars-bot/wheels/*.whl 2>&1 | tail -5`、cell C は `!pip install -q -e /kaggle/input/orbit-wars-bot/ 2>&1 | tail -5`。
5. **F2.5** cell D の subprocess は `stdout=PIPE` でログ取得し `/kaggle/working/runs/<RUN_ID>/train.log` に保存。stdout cap 100MB。
6. **F2.6** cell E は `data/output/models/imitation/<CASE>/runs/<RUN_ID>/` 配下の artifact を `/kaggle/working/runs/<RUN_ID>/` にコピー、`run.json` の `kaggle_kernel_meta` を埋める。
7. **F2.7** cell F (cleanup): `/kaggle/working/runs/<RUN_ID>/` のサイズが 18GB を超えていれば log を truncate (output size 上限対策)。

### F3. Pull / Promote / Status / Logs / Watch / Cost-report

3.1. **`dev/kaggle-kernel pull <run_id>`**:
   - `kaggle kernels output <slug>` で `/kaggle/working/` の中身を `data/output/models/imitation/<case>/runs/<run_id>/` に配置。
   - `dvc add data/output/models/imitation/<case>/runs/<run_id>/` → `dvc push`。
   - `run.json` を pretty-print 表示。

3.2. **`dev/kaggle-kernel promote <run_id> [--eval-results PATH]`**: `runpod_io.artifacts.run_meta.promote_to_canonical()` を共有 import で呼ぶ。

3.3. **`dev/kaggle-kernel status <run_id>`**: `kernels_status` の結果と launch.json と run.json (あれば) を 1 view にまとめて表示。

3.4. **`dev/kaggle-kernel ps`**: `kernels_list(user=...)` で active (queued/running) を表示。

3.5. **`dev/kaggle-kernel logs <run_id> [--tail N] [--grep PAT]`**: kernel 完了後、output に含まれる `train.log` を読み出して表示。`--tail` / `--grep` でフィルタ。

3.6. **`dev/kaggle-kernel watch <run_id> [--poll-interval 60] [--max-wait 36000]`**: 既存 kernel が complete / error になるまで polling、完了で desktop 通知。

3.7. **`dev/kaggle-kernel cost-report [--month YYYY-MM]`**: `runs/*/run.json` を全走査し `kaggle_kernel_meta != None` の run の `runtime_seconds` を月別で集計。`docs/experiment/kaggle_kernel_cost_report_<YYYY-MM>.md` に出力。

### F4. Dataset CLI: `dev/kaggle-kernel dataset {push|status|list}`

4.1. **`dataset push [--commit-sha SHA] [--force-new]`**: `bot/` snapshot を tar 化 + dataset-metadata.json 生成 → `dataset_create_version` (or `dataset_create_new` if `--force-new`).

4.2. **`dataset status [--slug SLUG]`**: dataset の processing 状態と最新 version の commit SHA を表示。

### F5. Auth: `bot/src/kaggle_kernel/auth.py`

5.1. `load_kaggle_creds()` は `KAGGLE_USERNAME` + `KAGGLE_KEY` env → `bot/.env` → `~/.kaggle/kaggle.json` の 3 段 fallback。
5.2. 失敗時は actionable な `CredentialsError` (上記 3 経路の設定方法を提示)。

### F6. train.py パッチ

6.1. `bot/pipeline/imitation/case{1,3,4,8,9}/training/train.py` の env 検出ブロックに `ORBIT_WARS_KAGGLE_KERNEL_SLUG` を追加、三 provider 同時セットは RuntimeError。
6.2. `kaggle_kernel_meta` を `ORBIT_WARS_KAGGLE_KERNEL_META` env (JSON 文字列) から `json.loads` で展開。

## Non-Functional Requirements

- **NFR1**: `dev/test-bot` グリーン維持。Kaggle API 呼び出しは fixture で全 mock。
- **NFR2**: secrets (KAGGLE_KEY) はファイルに hard-code せず env / `bot/.env` (gitignored) で扱う (`.claude/rules/security.md` 準拠)。
- **NFR3**: `run.json` の schema v1 を保ち、既存 vast/runpod run.json の round-trip は破壊しない。
- **NFR4**: `dev/kaggle-kernel train` 一発で **3 分以内に kernel push 完了** (実 GPU 起動は Kaggle 側 queue 次第)。
- **NFR5**: 同時 kernel 数を超えそうな場合の safety check (4 件以上で confirm)。

## Out of Scope

- Real-time log streaming (`tail`): Kaggle API は post-mortem のみ
- GPU offer marketplace 系 (`stock`, `--cloud-type`): Kaggle に該当機能なし
- Network volume CRUD: Kaggle に該当機能なし、Dataset で代替
- Internet OFF competition への対応: 本基盤は internet ON 前提、Phase 2 で検討
- Submit kernel の生成: 既存 `dev/submit` の責務
- 三 provider 横断の統合 dashboard: 別 feature
