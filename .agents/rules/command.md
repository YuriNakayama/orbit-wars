# Command Execution Rules

Conventions for running scripts and tooling in this repository. Prefer the wrappers under `dev/` over invoking `uv` / `dvc` / package managers directly: the wrappers `cd` into `bot/` and pin the right interpreter, so the same command works from any worktree.

## Top-level Commands

```bash
dev/setup             # Install dependencies (uv sync)
dev/format            # Code formatting (ruff)
dev/lint              # Static analysis (ruff + mypy)
dev/test-bot          # CI (format check → lint → type check → pytest)
dev/create-worktree   # Create git worktree with .env copy
dev/dvc               # DVC operations (setup / pull / repro / push / dag / add)
dev/vast              # Vast.ai GPU pod control (train / pull / promote / cost-report / volume)
dev/runpod            # RunPod GPU pod control (train / pull / promote / cost-report / volume)
```

## DVC Commands

```bash
dev/dvc setup                       # Configure local DVC (cache dir + AWS profile)
dev/dvc pull                        # Fetch real data from S3 remote
dev/dvc repro                       # Re-run pipeline on diffs
dev/dvc push                        # Upload artifacts to S3
dev/dvc dag                         # Stage dependency graph
dev/dvc add <path>                  # Track a path with DVC
dev/dvc <subcommand> [args...]      # Pass-through to `dvc <subcommand>`
```

`data/lake/selfplay/matches/` (selfplay runner output) and `data/lake/kaggle_episodes/matches/` (Kaggle scraper output) are tracked at the directory level via `dev/dvc add`. When the selfplay run produces new history, either pass `--dvc-add` to update automatically, or run `dev/dvc add data/lake/selfplay/matches` → `git add *.dvc` → `dev/dvc push` manually.

**Concurrent execution across multiple worktrees is discouraged**: the DVC cache at `/Users/user/project/orbit-wars/.dvc/cache` is shared between worktrees, so simultaneously running `dev/dvc repro` / `dev/dvc pull` / `dev/dvc add` may cause lock contention.

## Vast.ai GPU Training

```bash
# 1) commit & push, then launch on Vast
git push origin <branch>
dev/vast train <commit-sha> [--stage train_imitation_case1]

# 2) once finished, fetch locally
dev/vast pull <run_id>

# 3) if adopted, promote to canonical weights
dev/vast promote <run_id>

# Cost check
dev/vast cost-report --month 2026-04
```

Candidate weights are saved to `data/output/models/imitation/case1/runs/<run_id>/best.pt` and managed via DVC/S3. `policy/weights.pt` (the canonical Kaggle submit weights) is updated only when `dev/vast promote` runs. `VAST_API_KEY` is recorded in `bot/.env`. See [`docs/plans/vast-ai-basis/`](../../docs/plans/vast-ai-basis/) for details.

## RunPod GPU Training

RunPod 基盤は Vast.ai と並走するもう一つの GPU プロバイダ。Secure Cloud (T3/T4 DC + network volume 可) と Community Cloud (P2P, 安価) の 2 系統を `--cloud-type` で選べる。

```bash
# 1) commit & push, then launch on RunPod
git push origin <branch>
dev/runpod train <commit-sha> [--case case1] [--cloud-type SECURE|COMMUNITY|ALL]

# 2) once finished, fetch locally
dev/runpod pull <run_id> [--case case1]

# 3) if adopted, promote to canonical weights
dev/runpod promote <run_id> [--case case1] [--eval-results PATH]

# Cost check (RunPod 専用、vast とは別ファイル)
dev/runpod cost-report --month 2026-05

# Network volume 管理 (Secure Cloud 専用、Pod 作成時のみ attach 可能)
dev/runpod volume list
dev/runpod volume search [--data-center-id US-KS-2]
dev/runpod volume create <name> --data-center-id US-KS-2 [--size 15]

# 進捗確認 / 完了監視
dev/runpod ps                         # 起動中 pod 一覧 (launch.json と突合)
dev/runpod status <run_id>            # 単一 run の pod state + S3 marker + DVC 状況
dev/runpod summary <run_id>           # status / cost / metrics / artifacts を 1 画面集約

# ライブ tail (pod RUNNING 中のみ、SSH 経由、永続化なし)
dev/runpod tail <run_id> --source onstart  # /var/log/onstart.log を tail -F
dev/runpod tail <run_id> --source train    # 学習プロセス stdout のみ
dev/runpod tail <run_id> --source gpu      # nvidia-smi 10s サンプル

# 永続化済ログ (terminate 後でも S3 経由で参照可)
dev/runpod logs <run_id>              # S3 progress marker を timestamp 順に表示
dev/runpod logs <run_id> --source onstart  # /var/log/onstart.log 全文 (run_dir or S3 fallback)
dev/runpod logs <run_id> --tail 5     # 末尾のみ
dev/runpod logs <run_id> --grep done  # 行フィルタ

dev/runpod watch <run_id>             # 既存 pod の終了まで poll → 完了/失敗で desktop 通知

# 成果物取得 (DVC 失敗時の S3 fallback あり)
dev/runpod pull <run_id>              # auto: DVC → 失敗時 S3 artifacts へ自動切替
dev/runpod pull <run_id> --from s3    # 強制 S3 artifacts 経由
dev/runpod pull <run_id> --from dvc   # 強制 DVC 経由 (fallback なし)

# `dev/runpod train --watch` で起動と同時に監視も開始可能 (推奨)。
# 終了通知は macOS osascript / Linux notify-send / fallback stdout。
# 観測性の詳細は docs/plans/runpod-basis/06_observability.md
```

Vast.ai 基盤と同じ `data/output/models/imitation/case<N>/runs/<run_id>/` に成果物を保存し、DVC/S3 remote も共有。run.json には provider 別フィールド (`vast_*` / `runpod_*`) が記録され、両基盤の run を区別可能。`RUNPOD_API_KEY` は `backend/.env` に置き、key は <https://runpod.io/console/user/settings> で発行。デフォルト cost limit は $1.5/run (Vast の $1.0 より高め)。詳細は [`docs/plans/runpod-basis/`](../../docs/plans/runpod-basis/)。

両基盤の使い分け方針は [`docs/plans/runpod-basis/README.md`](../../docs/plans/runpod-basis/README.md) の「Vast.ai 基盤との使い分け」表を参照。

## Kaggle Submission Policy

Any real remote submission (`uv run python -m submit submit`, `dev/submit`, `kaggle competitions submit`, the `cd-kaggle-submit.yml` workflow_dispatch) is irreversible and consumes the daily 5-submission quota (note: `SubmissionStatus.ERROR` does NOT count against the quota — validation failures can be retried immediately). Always obtain explicit user approval immediately before executing, showing the case / message / mode to be submitted. Dry-run, archive build, and read-only history checks do NOT require approval. Prior approval covers only that single submission and does not extend to later submissions or auto-mode / autonomous loops.

## Direct `uv` / `dvc` Usage

Only fall back to running `uv run --directory bot ...` or `dvc ...` directly when no wrapper covers the case. If the same direct invocation appears more than once, add it as a subcommand under the relevant `dev/` script instead of duplicating it across docs.
