---
paths:
  - "dev/**"
---

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
dev/kaggle            # Kaggle Notebook free-tier GPU (train / pull / promote / dataset / cost-report)
```

## Long-running Training Checkpoint Policy

Long-running training (RunPod / Kaggle / Vast runs expected to exceed 1h) **must upload intermediate artifacts such as best.pt to S3 (or the DVC remote) on every iter, immediately**. A Kaggle Kernel only commits `/kaggle/working` to its output on a clean finish, so intermediate weights are discarded on timeout / ERROR (and lost to preemption on RunPod). Always pass an env var such as `ORBIT_WARS_BEST_S3_PREFIX` to the train script and adopt the implementation pattern of calling `s3.upload_file()` on every new best. When adding a new training platform, satisfy this convention before putting it into production use.

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

RunPod is a second GPU provider running alongside Vast.ai. You can choose between two tiers with `--cloud-type`: Secure Cloud (T3/T4 DC + network volume support) and Community Cloud (P2P, cheaper).

```bash
# 1) commit & push, then launch on RunPod
git push origin <branch>
dev/runpod train <commit-sha> [--case case1] [--cloud-type SECURE|COMMUNITY|ALL]

# 2) once finished, fetch locally
dev/runpod pull <run_id> [--case case1]

# 3) if adopted, promote to canonical weights
dev/runpod promote <run_id> [--case case1] [--eval-results PATH]

# Cost check (RunPod-specific, separate file from vast)
dev/runpod cost-report --month 2026-05

# Network volume management (Secure Cloud only, can only be attached at Pod creation)
dev/runpod volume list
dev/runpod volume search [--data-center-id US-KS-2]
dev/runpod volume create <name> --data-center-id US-KS-2 [--size 15]

# Progress check / completion monitoring
dev/runpod ps                         # list running pods (reconciled against launch.json)
dev/runpod status <run_id>            # single run's pod state + S3 marker + DVC status
dev/runpod summary <run_id>           # status / cost / metrics / artifacts in one screen

# Live tail (only while pod is RUNNING, over SSH, not persisted)
dev/runpod tail <run_id> --source onstart  # tail -F /var/log/onstart.log
dev/runpod tail <run_id> --source train    # training process stdout only
dev/runpod tail <run_id> --source gpu      # nvidia-smi 10s samples

# Persisted logs (viewable via S3 even after terminate)
dev/runpod logs <run_id>              # show S3 progress markers in timestamp order
dev/runpod logs <run_id> --source onstart  # full /var/log/onstart.log (run_dir or S3 fallback)
dev/runpod logs <run_id> --tail 5     # tail only
dev/runpod logs <run_id> --grep done  # line filter

dev/runpod watch <run_id>             # poll an existing pod until exit → desktop notify on success/failure

# Artifact retrieval (S3 fallback when DVC fails)
dev/runpod pull <run_id>              # auto: DVC → falls back to S3 artifacts on failure
dev/runpod pull <run_id> --from s3    # force S3 artifacts path
dev/runpod pull <run_id> --from dvc   # force DVC path (no fallback)

# `dev/runpod train --watch` can start monitoring at launch time (recommended).
# Completion notifications use macOS osascript / Linux notify-send / fallback stdout.
# Observability details: docs/plans/runpod-basis/06_observability.md
```

Artifacts are saved under the same `data/output/models/imitation/case<N>/runs/<run_id>/` as the Vast.ai platform, and the DVC/S3 remote is shared. `run.json` records provider-specific fields (`vast_*` / `runpod_*`), so runs from both platforms are distinguishable. `RUNPOD_API_KEY` goes in `backend/.env`; the key is issued at <https://runpod.io/console/user/settings>. The default cost limit is $1.5/run (higher than Vast's $1.0). See [`docs/plans/runpod-basis/`](../../docs/plans/runpod-basis/) for details.

For the policy on when to use which platform, see the "Choosing between Vast.ai and RunPod" table in [`docs/plans/runpod-basis/README.md`](../../docs/plans/runpod-basis/README.md).

### Interactive Mode (dev / debug pods)

`dev/runpod dev` reserves a pod in **interactive mode** and keeps it alive with `sleep infinity`. It disables all of auto-cleanup / the 8h timeout guard / `trap cleanup_destroy EXIT`, so you can repeatedly edit code, re-run, and debug over an SSH connection. You must terminate it explicitly with `dev/runpod destroy <run_id>` (leaving it running keeps incurring charges).

```bash
# Launch (the commit must already be pushed to origin)
dev/runpod dev <commit-sha> [--case caseN] [--cloud-type SECURE|COMMUNITY|ALL]

# Status check (SSH-ready once 50_interactive_ready appears)
dev/runpod status <run_id> --case caseN

# SSH connection (proxy=ssh.runpod.io by default; direct=TCP/22 public port also possible)
dev/runpod ssh <run_id> [--case caseN] [--via proxy|direct] [--key PATH] [--exec "<cmd>"]

# Code sync (over rsync, bot/ only; excludes .venv / data / __pycache__ etc.)
dev/runpod sync <run_id> [--case caseN] --push [--dry-run] [--delete]
dev/runpod sync <run_id> [--case caseN] --pull

# Explicit terminate (mandatory in interactive mode)
dev/runpod destroy <run_id> [--case caseN] [-y]
```

For a comparison of oneshot mode (`dev/runpod train`) vs interactive mode (`dev/runpod dev`) and how to register a key for proxy SSH, see [`docs/plans/runpod-basis/07_interactive_mode.md`](../../docs/plans/runpod-basis/07_interactive_mode.md). `dev/runpod ps` shows interactive pods in yellow and prints a destroy reminder.

## Kaggle Kernel GPU Training (Free Tier)

A third GPU training platform alongside Vast.ai / RunPod. It uses the **free GPU tier (T4x2 / P100, 30h/week)** of Kaggle Notebooks (Save & Run All batch execution) to run training at zero cost. It has a 9h GPU cap and a limit of ~5 concurrent kernels; unsuitable for long RL runs but good for small imitation cases.

```bash
# 0) (first time only) Add the Kaggle API key to bot/.env
#    https://www.kaggle.com/settings → Create New API Token to get kaggle.json, then
#    add KAGGLE_USERNAME=<your-username> and KAGGLE_KEY=<your-key> to bot/.env
# 1) (first time only) Upload bot/ as a Kaggle Dataset
dev/kaggle dataset push --commit-sha "$(git rev-parse HEAD)"

# 2) commit & push, then launch on Kaggle
git push origin <branch>
dev/kaggle train "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2 --watch
#   → push the bot/ snapshot as a new dataset version
#   → auto-generate the notebook → push the kernel
#   → --watch polls QUEUED → RUNNING → COMPLETE / ERROR

# 3) pull artifacts (kaggle kernels output → local dvc add)
dev/kaggle pull <run_id> --case case1

# 4) if adopted, promote to canonical weights (same as vast/runpod)
dev/kaggle promote <run_id> --case case1 [--eval-results PATH]

# 5) monthly free GPU hours report (cost is 0)
dev/kaggle cost-report --month 2026-05

# Progress check / logs
dev/kaggle ps                            # list active kernels
dev/kaggle status <run_id>               # single run's launch + kernel status + run.json
dev/kaggle watch <run_id>                # poll until finished, desktop notify on completion
dev/kaggle logs <run_id> [--tail N]      # train.log of a finished kernel (requires prior pull)

# Dataset management
dev/kaggle dataset push --label "<note>"  # record the commit SHA in version_notes
dev/kaggle dataset status                 # current dataset's processing state
```

Artifacts are saved under the same `data/output/models/imitation/case<N>/runs/<run_id>/` as Vast.ai / RunPod, and the DVC/S3 remote is shared. The `kaggle_kernel_meta` field in `run.json` distinguishes runs that went through Kaggle. `KAGGLE_USERNAME` / `KAGGLE_KEY` go in `bot/.env`; the key is issued at <https://www.kaggle.com/settings>.

A Kaggle Kernel is for **training**, not the Kaggle competition submit kernel (submission is the responsibility of `dev/submit`). See [`docs/plans/kaggle-kernel-basis/`](../../docs/plans/kaggle-kernel-basis/) for details.

### Interactive Mode (Kaggle Notebook resident via a sleep loop + S3 command channel)

Since Kaggle has no SSH, `dev/kaggle dev` pushes a **sleep-loop notebook that uses S3 as a bidirectional channel**. When Claude (local) puts a command in the S3 inbox, the kernel picks it up, executes it, and writes the result back to the outbox. Functionally equivalent to RunPod's `dev/ssh/sync/destroy`. The core point: **even if the kernel dies on ERROR / OOM, a heartbeat + the last outbox remain in S3, so you can grasp the situation just before death without SSH.**

```bash
# 1) launch an interactive kernel (equivalent to RunPod dev)
git push origin <branch>
dev/kaggle dev "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2

# 2) check heartbeat (state=ready / running / idle / shutdown / voluntary_exit)
dev/kaggle info <run_id>

# 3) run arbitrary bash on the kernel (via S3, equivalent to ssh-exec)
dev/kaggle exec <run_id> -- python -c "import torch; print(torch.cuda.is_available())"
dev/kaggle exec <run_id> --cwd /tmp/orbit-wars-repo/bot -- pytest tests/unit/

# 4) transfer a local file to the kernel (equivalent to rsync push)
dev/kaggle sync <run_id> --file bot/pipeline/imitation/case1/training/train.py

# 5) explicit shutdown (Kaggle auto-stops on the next quota cycle; for immediate stop use the Web UI)
dev/kaggle destroy <run_id> -y
```

Constraints: Internet ON is required (for S3 access, so it cannot be reused for a submit kernel); AWS creds are also needed on the kernel side (bundle `bot/.env` into the dataset or register it as a Kaggle Secret); the kernel is force-stopped past Kaggle's 9h cap (it is recommended to bring the voluntary exit forward with `--max-idle-minutes`).

## Kaggle Submission Policy

Any real remote submission (`uv run python -m submit submit`, `dev/submit`, `kaggle competitions submit`, the `cd-kaggle-submit.yml` workflow_dispatch) is irreversible and consumes the daily 5-submission quota (note: `SubmissionStatus.ERROR` does NOT count against the quota — validation failures can be retried immediately). Always obtain explicit user approval immediately before executing, showing the case / message / mode to be submitted. Dry-run, archive build, and read-only history checks do NOT require approval. Prior approval covers only that single submission and does not extend to later submissions or auto-mode / autonomous loops.

## Direct `uv` / `dvc` Usage

Only fall back to running `uv run --directory bot ...` or `dvc ...` directly when no wrapper covers the case. If the same direct invocation appears more than once, add it as a subcommand under the relevant `dev/` script instead of duplicating it across docs.
