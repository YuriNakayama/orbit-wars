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
dev/vast              # Remote GPU server control
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

## Kaggle Submission Policy

Any real remote submission (`uv run python -m submit submit`, `dev/submit`, `kaggle competitions submit`, the `cd-kaggle-submit.yml` workflow_dispatch) is irreversible and consumes the daily 5-submission quota (note: `SubmissionStatus.ERROR` does NOT count against the quota — validation failures can be retried immediately). Always obtain explicit user approval immediately before executing, showing the case / message / mode to be submitted. Dry-run, archive build, and read-only history checks do NOT require approval. Prior approval covers only that single submission and does not extend to later submissions or auto-mode / autonomous loops.

## Direct `uv` / `dvc` Usage

Only fall back to running `uv run --directory bot ...` or `dvc ...` directly when no wrapper covers the case. If the same direct invocation appears more than once, add it as a subcommand under the relevant `dev/` script instead of duplicating it across docs.
