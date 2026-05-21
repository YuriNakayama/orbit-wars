# kaggle-kernel-basis — Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Developer Laptop                                                        │
│                                                                         │
│   $ dev/kaggle-kernel train <sha> --case case1 --watch                  │
│           │                                                             │
│           ▼                                                             │
│   ┌──────────────────────────────┐                                      │
│   │ bot/src/kaggle_kernel/cli/   │                                      │
│   │   app.py (Typer)             │                                      │
│   │  1. git push verification    │                                      │
│   │  2. load_kaggle_creds()      │                                      │
│   │  3. dataset.api: bump version│                                      │
│   │     (if commit SHA new)      │                                      │
│   │  4. kernel.template: render  │                                      │
│   │     main.ipynb               │                                      │
│   │  5. kernel.runner: push +    │                                      │
│   │     poll status              │                                      │
│   │  6. artifacts.output: pull → │                                      │
│   │     run_dir + dvc add        │                                      │
│   └──────────────────────────────┘                                      │
│                                                                         │
└──────────┬──────────────────────────────────────────┬───────────────────┘
           │ (a) dataset_create_version             │ (b) kernels_push
           ▼                                          ▼
┌─────────────────────────┐               ┌────────────────────────────────┐
│ Kaggle Datasets         │               │ Kaggle Notebooks (Kernel)      │
│   <user>/orbit-wars-bot │ ◀── attach ── │  /kaggle/input/orbit-wars-bot/ │
│   - bot/ snapshot       │               │  ┌──────────────────────────┐  │
│   - wheels/ (rust)      │               │  │ cell A: env setup        │  │
│   - dataset metadata    │               │  │ cell B: pip install wheel│  │
└─────────────────────────┘               │  │ cell C: pip install -e   │  │
                                          │  │ cell D: python -m train  │  │
                                          │  │ cell E: copy artifacts → │  │
                                          │  │   /kaggle/working/runs/  │  │
                                          │  │     <run_id>/            │  │
                                          │  └──────────────────────────┘  │
                                          │             │                  │
                                          │             ▼                  │
                                          │  /kaggle/working/ (output)     │
                                          └─────────────┬──────────────────┘
                                                        │ (c) kernels_output
                                                        ▼
                                          ┌────────────────────────────────┐
                                          │ Developer Laptop               │
                                          │  data/output/models/imitation/ │
                                          │    case1/runs/<run_id>/        │
                                          │  ├── best.pt                   │
                                          │  ├── metrics.json              │
                                          │  ├── run.json (kaggle_kernel_  │
                                          │  │   meta 含む)                │
                                          │  └── train.log                 │
                                          │                                │
                                          │  $ dvc add ... && dvc push     │
                                          │       │                        │
                                          │       ▼                        │
                                          │   S3 (orbit-wars-dvc-...)      │
                                          └────────────────────────────────┘
```

## Directory Layout

```
bot/src/kaggle_kernel/
├── __init__.py
├── __main__.py                    # `python -m kaggle_kernel` entrypoint
├── auth.py                        # KAGGLE_USERNAME/KAGGLE_KEY 解決 (3 段 fallback)
├── config/
│   └── __init__.py                # runpod_io.config.cases を re-export + kaggle 固有 default
├── dataset/
│   ├── __init__.py
│   ├── builder.py                 # bot/ → snapshot tar 作成、除外規則
│   ├── api.py                     # kaggle datasets create/version (KaggleApi wrapper)
│   └── metadata.py                # dataset-metadata.json 生成
├── kernel/
│   ├── __init__.py
│   ├── template.py                # notebook (.ipynb) テンプレ render
│   ├── runner.py                  # KaggleApi().kernels_push + status polling
│   └── state.py                   # KernelStatus enum + helpers
├── artifacts/
│   ├── __init__.py
│   ├── launch.py                  # launch.json (kernel slug/version/dataset_version)
│   ├── output.py                  # kernels_output → run_dir 配置 → dvc add
│   └── cost.py                    # 月次 free GPU hour 集計
└── cli/
    ├── __init__.py
    └── app.py                     # Typer subcommands: train/pull/promote/status/ps/logs/watch/cost-report/dataset

bot/tests/src/kaggle_kernel/
├── __init__.py
├── conftest.py                    # KaggleApi mock fixture
├── test_auth.py
├── test_dataset_builder.py
├── test_dataset_api.py
├── test_kernel_template.py
├── test_kernel_runner.py
├── test_artifacts_output.py
├── test_artifacts_cost.py
└── test_cli_train.py              # train サブの統合テスト (kaggle.api 全 mock)

dev/kaggle-kernel                  # 3 行 bash wrapper

docs/plans/kaggle-kernel-basis/    # 本ディレクトリ
```

## Data Model

### `RunMetadata.kaggle_kernel_meta`

`bot/src/vast/run_meta.py` の `RunMetadata` dataclass に追加する optional field:

```python
@dataclass(frozen=True)
class RunMetadata:
    schema_version: int = 1
    # ... 既存 fields ...
    vast_instance_id: str | None = None
    vast_offer_snapshot: dict[str, Any] | None = None
    runpod_pod_id: str | None = None
    runpod_offer_snapshot: dict[str, Any] | None = None
    # NEW
    kaggle_kernel_meta: dict[str, Any] | None = None
```

`kaggle_kernel_meta` の中身 (JSON shape):
```json
{
  "kernel_slug": "username/orbit-wars-case1-20260520-abc1234",
  "kernel_version": 3,
  "dataset_slug": "username/orbit-wars-bot",
  "dataset_version": "v17",
  "accelerator": "gpu-t4x2",
  "runtime_seconds": 1820,
  "internet_enabled": true,
  "free_gpu_hours_remaining_at_start": 24.5,
  "queued_at": "2026-05-20T10:30:00Z",
  "started_at": "2026-05-20T10:31:42Z",
  "completed_at": "2026-05-20T11:02:02Z"
}
```

### `launch.json` (per-run launch metadata, written by `dev/kaggle-kernel train`)

```json
{
  "run_id": "20260520-103000__feature-x__abc1234__seed0",
  "case": "case1",
  "commit_sha": "abc1234...",
  "branch": "feature-x",
  "kernel_slug": "username/orbit-wars-case1-20260520-abc1234",
  "kernel_version": 3,
  "dataset_slug": "username/orbit-wars-bot",
  "dataset_version": "v17",
  "accelerator": "gpu-t4x2",
  "enable_internet": true,
  "started_at": "2026-05-20T10:30:00Z",
  "estimated_free_gpu_hours_used": 0.5
}
```

保存先: `data/output/models/imitation/case<N>/runs/<run_id>/launch.json` (provider 中立で同じ場所)。

## Module Dependencies

```
cli/app.py
  ├── auth.load_kaggle_creds
  ├── dataset/
  │   ├── builder.build_snapshot
  │   ├── metadata.write_dataset_metadata
  │   └── api.{push_dataset_version, dataset_status}
  ├── kernel/
  │   ├── template.render_notebook
  │   ├── runner.{push_kernel, poll_status}
  │   └── state.KernelStatus
  ├── artifacts/
  │   ├── launch.{write_launch_json, read_launch_json}
  │   ├── output.{pull_outputs, place_into_run_dir, dvc_add}
  │   └── cost.aggregate_runs
  ├── config (← runpod_io.config.cases を re-export)
  └── runpod_io.notify (← 共有 import)
       runpod_io.artifacts.run_meta.promote_to_canonical (← 共有 import)

dataset/api.py → KaggleApi (kaggle SDK)
kernel/runner.py → KaggleApi
artifacts/output.py → KaggleApi
```

`runpod_io` からの import 経路:
- `runpod_io.notify` — desktop 通知ヘルパ (macOS osascript / linux notify-send / fallback)
- `runpod_io.artifacts.run_meta.promote_to_canonical` — `policy/weights.pt` への昇格
- `runpod_io.config.cases` — case 別 train_module / canonical weights path

これらは provider 非依存なロジックなので、本基盤からも再利用する (DRY)。将来必要なら `bot/src/cloud_common/` 等に切り出し。

## Notebook Template Structure

`bot/src/kaggle_kernel/kernel/template.py` が生成する `main.ipynb` の JSON 概略:

```json
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# cell A: env setup\n",
        "import os, json\n",
        "os.environ.update({\n",
        "  'ORBIT_WARS_RUN_ID': '<RUN_ID>',\n",
        "  'ORBIT_WARS_GIT_SHA': '<COMMIT_SHA>',\n",
        "  'ORBIT_WARS_GIT_BRANCH': '<BRANCH>',\n",
        "  'ORBIT_WARS_CASE': '<CASE>',\n",
        "  'ORBIT_WARS_KAGGLE_KERNEL_SLUG': '<KERNEL_SLUG>',\n",
        "  'ORBIT_WARS_KAGGLE_KERNEL_VERSION': '<KERNEL_VERSION>',\n",
        "  'ORBIT_WARS_KAGGLE_ACCELERATOR': '<ACCELERATOR>',\n",
        "  'ORBIT_WARS_KAGGLE_KERNEL_META': '<KAGGLE_META_JSON>',\n",
        "  'ORBIT_WARS_RUN_DIR': '/kaggle/working/runs/<RUN_ID>',\n",
        "})\n",
        "os.makedirs(os.environ['ORBIT_WARS_RUN_DIR'], exist_ok=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": ["!pip install -q /kaggle/input/orbit-wars-bot/wheels/*.whl 2>&1 | tail -3"]
    },
    {
      "cell_type": "code",
      "source": ["!pip install -q -e /kaggle/input/orbit-wars-bot/ 2>&1 | tail -3"]
    },
    {
      "cell_type": "code",
      "source": [
        "# cell D: train\n",
        "import subprocess, sys, os\n",
        "log_path = os.path.join(os.environ['ORBIT_WARS_RUN_DIR'], 'train.log')\n",
        "with open(log_path, 'w') as logf:\n",
        "  proc = subprocess.run(\n",
        "    [sys.executable, '-m', '<TRAIN_MODULE>', '<CONFIG_ARG>'],\n",
        "    stdout=logf, stderr=subprocess.STDOUT, check=True,\n",
        "  )\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# cell E: collect artifacts\n",
        "import shutil, glob\n",
        "src = f\"/kaggle/working/data/output/models/imitation/<CASE>/runs/<RUN_ID>/\"\n",
        "dst = os.environ['ORBIT_WARS_RUN_DIR']\n",
        "if os.path.isdir(src):\n",
        "  for f in glob.glob(os.path.join(src, '*')):\n",
        "    shutil.copy2(f, dst)\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

## Concurrency Model

- Kaggle 側で **active kernel 数 ≤ 5** の制約あり。`dev/kaggle-kernel ps` で現状確認、`train` 起動時に 4 件以上で confirm。
- 同時 `train` 起動は技術的に可能だが、無料 GPU quota (週 30h) を考慮し原則 1 件ずつ運用。
- DVC cache は worktree 共有 (既存運用と同じ) — `dev/kaggle-kernel pull` 同時実行は控える運用ルール。

## Failure Modes

| Layer | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| Auth | KAGGLE_KEY 不正 | `KaggleApi().authenticate()` で 403 | `CredentialsError` を raise、3 経路の設定方法を提示 |
| Dataset push | quota 超過 / SHA 重複 | `dataset_create_version` 戻り値 | retry 1 回、それでもダメなら `--dataset-bump-only` を提案 |
| Kernel push | metadata 不備 / Internet 制約違反 | API 400 | エラー詳細を表示、`kernel-metadata.json` の dump を表示 |
| Kernel run | 9h 超過 / OOM / import 失敗 | status=`error` + `failureMessage` | `dev/kaggle-kernel logs <run_id>` で原因確認、再 push |
| Output pull | kernel 未完了 / 20GB 超過 | `kernels_output` で 404 | status 確認、cell E で log truncate されているか確認 |
| dvc add | local disk full / .dvc 競合 | `dvc add` exit != 0 | 既存運用と同じ復旧 (空き確保、`dvc gc` 等) |

## Security

- `KAGGLE_KEY` は `bot/.env` (gitignored) に置く運用。notebook 本体には絶対に埋め込まない。
- `KaggleApi().authenticate()` は env → `~/.kaggle/kaggle.json` の自動解決経路を持つため、本基盤の `auth.py` は env を優先しつつ `kaggle.json` fallback も許容する。
- Kaggle Dataset の `isPrivate=true` で公開を防ぐ。
- AWS credentials は **Kaggle 側に渡さない** (A1 設計: output 経由でローカルから dvc push)。
- `.claude/rules/security.md` 準拠。
