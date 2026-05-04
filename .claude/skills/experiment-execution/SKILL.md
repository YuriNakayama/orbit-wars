---
name: experiment-execution
description: >
  Execution runner for Orbit Wars experiments under `backend/pipeline/`. Drives
  the full cycle directly in the main session (no subagent handoff): capture
  the hypothesis → write `docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/plan.md`
  → implement / extend the case under `backend/pipeline/{rulebase|imitation|reinforce}/case<N>/`
  → run `dev/test-backend` for local bug detection → push the commit and launch
  GPU training on RunPod via `dev/runpod train` → return control to the user
  while the run is in flight so they can ask for status or change direction →
  evaluate on local match outcomes and write `result.md`. Use whenever the user
  types `/experiment-execution`, or asks to run / execute / iterate / kick off an
  experiment, train a new model, launch a RunPod run, propose a new case, or
  write up an experiment result — even if they don't explicitly say "execute",
  phrases like "imitation/case1 で dropout を試したい",
  "rulebase/case2 を改良して回したい", "runpod で学習を回して結果まとめて",
  "新しい case を切って学習させたい", "plan.md に従って実装して回して" all
  count. Don't trigger this skill for hypothesis-only discussion / planning
  conversations (use `experiment-plan`), interactive result interpretation /
  follow-up analysis after a run finishes (use `experiment-analysis`),
  read-only code review, plain bug fixes, or Kaggle submission requests.
---

# Experiment Execution Skill (Orbit Wars)

This skill is the front door for any experiment that lives under `backend/pipeline/`. It translates a user's intent ("imitation/case1 で dropout を上げて回したい") into the concrete steps of an experiment lifecycle: plan.md → case implementation → pytest → RunPod launch → progress reporting → result.md.

The skill runs **inline in the main session** rather than delegating to a subagent. This is intentional: experiments are long-running, and the user must be able to ask "今どこまで進んだ?", "やっぱり dropout じゃなく lr 変えて" or "学習止めて別の case にして" while the work is in flight. A subagent handoff would block those interjections until the agent returns. Running inline keeps the conversation and the work in the same context.

## When this skill is in charge

- The user typed `/experiment-execution` (explicit trigger), **or**
- The user asked, in plain language, for any of:
  - a new experiment / case (`新しい case を切って…`, `RL の baseline を回したい`)
  - an iteration of an existing case (`imitation/case1 を改良して再学習`, `case4 の harass policy を強くして…`)
  - a RunPod training run (`runpod で学習回して`, `GPU で学習させたい`)
  - writing up the result of a recent run (`さっきの run 結果を docs にまとめて`)
- The user did **not** already pre-specify the entire pipeline themselves (e.g. "just run `dev/runpod train abc123`" — at that point, just run it).

If the user is asking for **read-only review**, **dead-code cleanup**, **Kaggle submission**, or **infra-only work**, redirect to the matching skill/agent (`code-review`, `python-review`, the submission flow under `dev/submit`, etc.) instead. Experiments are about hypothesis → trained model → result, not about polishing existing code.

## What "an experiment" means in this repo

A single experiment is bounded by:

- **Family** — `imitation` / `rulebase` / `reinforce`. Determined by which agent type the change targets.
- **Case** — `case<N>` under `backend/pipeline/<family>/`. New case vs. extending an existing one is a real fork: extending preserves comparable baselines, new cases are warranted only when the hypothesis genuinely doesn't fit the existing structure.
- **Hypothesis** — one sentence on the change being tested + the success metric (typically a local-match win rate against a named opponent, ≥300 episodes for fragile claims).
- **Compute target** — local CPU only, or RunPod GPU. Anything beyond a few minutes of training goes on RunPod.

Read `.claude/rules/docs.md` for `docs/experiment/...` naming and `.claude/rules/backend/pipeline.md` for case-directory submit constraints.

## Skill flow

The skill runs six phases. Each phase is a clean checkpoint where the user can interject — after every phase boundary, give a short status update (≤ 3 lines) before moving on, so the user has a chance to redirect.

### Phase 1 — Clarify the hypothesis

Before writing anything, you must know:

1. **Agent family** (`imitation` / `rulebase` / `reinforce`) and **target case number** (`case1`, `case4`, ...). New case vs. extending an existing one is a critical fork.
2. **Hypothesis / goal** — one sentence on what change is being tested and what success looks like (win-rate uplift, loss curve, etc.).
3. **Compute target** — local CPU only, or RunPod GPU. Default to RunPod for any model training that exceeds a few minutes.
4. **Evaluation method** — self-play vs. which baseline, episode count, threshold. Recall the project rule: Kaggle publicScore is unreliable; use local match outcomes only.
5. **Scope of this skill invocation** — full pipeline (plan → impl → test → runpod → result), partial (e.g. plan.md only, or "just write up the result of run xyz"), or "dry-run plan only".

Use `AskUserQuestion` to fill any unresolved field. Aim for **at most 1–2 questions**; skip what's already implied. If the user already said `dry-run のプランだけ` or `plan.md だけ書いて`, do not ask about RunPod — they've already scoped it down.

If the user asks for a "new experiment" without specifying a case number, scan `backend/pipeline/<family>/` and propose the next free `caseN` slot.

**GPU spend is real money.** First RunPod launch in a session needs explicit user opt-in with a one-line cost expectation. Subsequent launches in the same conversation under the same approved cost envelope do not need re-confirmation.

### Phase 2 — Quick reality check

Run these in parallel via `Bash`:

```bash
git status --short                      # is the working tree clean enough to push?
git rev-parse --abbrev-ref HEAD         # which branch will dev/runpod train pin from?
git rev-parse HEAD                      # SHA to record in plan.md
ls backend/pipeline/<family>/           # which case numbers already exist?
ls docs/experiment/<family>/            # is there a matching directory for this hypothesis (= iteration)?
```

If `git status` is dirty and the user wants a RunPod launch, surface that and decide together: commit first, stash, or proceed without RunPod. Do NOT silently launch from a dirty tree.

### Phase 3 — Locate or create the experiment directory and write `plan.md`

Follow `.claude/rules/docs.md` exactly:

```
docs/experiment/{imitation|rulebase|reinforce}/{yyyymmdd}_case{N}_{topic}/{plan|result|...}.md
```

- **Date**: today's date in `yyyymmdd` (use the `currentDate` injected into the session, not `date` from the shell, which can be off in some sandboxes).
- **`{topic}`**: short snake_case noun phrase derived from the hypothesis (e.g. `dropout_sweep`, `harass_policy`, `pure_il_cycle`).
- **Iterations**: if a directory for the same hypothesis already exists, do NOT create a new one. Add `iterN_plan.md` / `iterN_result.md` inside the existing directory. If the existing directory has a plain `plan.md`/`result.md` and you're starting iter2, rename the originals to `iter1_*.md` first (confirm the rename with the user).

Write `plan.md` with at minimum:

- **Hypothesis** — what's being tested, why it should work.
- **Scope** — files to add/change under `backend/pipeline/<family>/case<N>/`, dataset/feature changes, hyperparameters.
- **Implementation steps** — short numbered list with file paths.
- **Local validation** — which `pytest` paths cover the change.
- **Remote training** — RunPod case identifier (e.g. `--case case1`), expected duration, checkpoint path.
- **Evaluation** — opponents, episode count, primary metric, decision threshold.
- **Risks / known unknowns**.

Keep `plan.md` concise (one screen is ideal). Verbose plans rot.

If the user scoped the request to `plan.md only`, **stop here** and report the path. Do not proceed to Phase 4.

If the hypothesis itself is structurally complex (genuinely new architecture, not a parameter sweep), you may delegate the design pass to `Task(subagent_type="planner", ...)` and fold its output into `plan.md`. For routine sweeps, write `plan.md` yourself.

After writing `plan.md`, give a short status update and proceed to Phase 4.

### Phase 4 — Implement the case

You must comply with `.claude/rules/backend/pipeline.md`. The non-negotiable rules:

- `pipeline/<family>/case<N>/main.py` exists, exposes `agent(obs)`, and uses `sys.path.insert(0, str(Path.cwd()))` (NOT `__file__`).
- All intra-package imports inside `case<N>/` are **relative**; never write `from pipeline.<family>.case<N>....` inside the case package.
- New case → register it in `backend/src/dataset/selfplay/agents.py` (`AGENT_REGISTRY`).
- Development-only directories (`evaluation/`, `notebook/`, `eda/`, `training/`, `configs/`) must be in `backend/pipeline/.submitignore`.

For existing-case extension, prefer **adding a new module** (`policy_v2.py`, `features_v3.py`) over rewriting the current one in place — it preserves a working baseline to compare against.

After implementing, before any remote run:

1. Confirm the local import works:
   ```bash
   cd backend && uv run python -c "from pipeline.<family>.case<N>.baseline.agent import agent; print(agent)"
   ```
2. Run the relevant tests:
   ```bash
   dev/test-backend                                          # full CI gate
   # or, if iterating fast:
   uv run --directory backend pytest tests/pipeline/<family>/case<N> -x
   ```
3. For changes to `main.py` / submit shape, dry-run the validator:
   ```bash
   uv run --directory backend python -m submit submit <family>/case<N> --dry-run -m "dry-run"
   ```

If `dev/test-backend` fails, do not proceed to remote training. Either fix the failure yourself, or — if it's clearly a typing/lint regression unrelated to the experiment logic — delegate to `python-build-resolver` via `Task`. Report the failure and resolution to the user.

After tests pass, give a short status update and proceed to Phase 5.

### Phase 5 — Launch remote training (RunPod)

Skip this phase entirely if the user scoped the request to local-only or plan-only.

Follow `.claude/rules/command.md`:

```bash
git push origin <branch>                          # commit + push first; runpod trains from a SHA
dev/runpod train <commit-sha> [--case <caseN>] [--cloud-type SECURE|COMMUNITY|ALL]
```

Pre-flight checks before invoking `dev/runpod train`:

- `git status` is clean (no uncommitted changes that won't make it into the run).
- The commit being trained from is pushed to the remote.
- The user has approved the run if it is the first time GPU budget is being spent in this conversation.
- **Never bypass interactive cost-confirmation prompts.** `dev/runpod train` may prompt for `[y/N]` when estimated cost exceeds the configured cap or for the first launch. Do NOT use `yes |`, `echo y |`, `--yes`, `printf 'y\n' |`, `expect`, or any other auto-answer mechanism to skip that prompt. Such mechanisms suppress the safety gate even when the parent session approved the *scope* — the per-command confirmation is its own gate. If the prompt blocks, stop and either (a) ask the user to type `y` manually via `! dev/runpod train ...`, (b) re-confirm the exact cost figure with the user before proceeding, or (c) raise the issue and let the user decide.

Capture the `run_id` printed by `dev/runpod train`. Save it (and the commit SHA, case, start time) — you'll need it for status checks and for promoting weights later.

Immediately after launch, report to the user (in Japanese, ≤ 3 lines):
- `run_id`, branch SHA, case
- Rough ETA
- "途中経過は『status 確認して』で見れます。10分おきに cron で自動監視し、失敗を検知したら停止 → 修正 → 再起動を試みます。"

Then proceed to **Phase 6.5** to set up the cron-driven health check.

Then **return control to the user** — do not busy-wait. The conversation stays in the main session, so any follow-up the user types ("status", "やめて", "lr 変えて再開") arrives directly without going through a subagent.

### Phase 6 — Status checks while in flight (handle inline as the user asks)

Because this skill runs inline, the user can interleave status requests with other conversation. Handle each request when it arrives, then return control again. Patterns:

- **"今 どこ?" / "進捗 教えて"** → run `dev/runpod status <run_id>` and report status, latest train/val metric, ETA delta, any errors. ≤ 3 lines.
- **"ログ 見せて"** → `dev/runpod tail <run_id> --source train` (live, while pod is RUNNING) or `dev/runpod logs <run_id>` (post-mortem from S3).
- **"成果物 まだ取れる?"** → `dev/runpod pull <run_id>` is safe before completion — it returns whatever artifacts exist so far.
- **"やめたい" / "途中で止めて"** — surface that this terminates the pod and forfeits remaining cost; confirm before any destructive call.
- **"plan を変えたい" / "lr 変えて再学習"** — treat as a new iteration. Do NOT mutate the in-flight run; instead, write `iterN+1_plan.md`, implement, and launch a fresh `dev/runpod train` after the current run finishes (or after the user decides to terminate).

Never claim a run "succeeded" until artifacts (`best.pt`, metrics JSON) are actually pulled and inspected locally.

If the user wants asynchronous notification when the run finishes, they can use `dev/runpod watch <run_id>` (desktop notification) or `dev/runpod train --watch` at launch time. The skill itself does NOT poll in a loop.

#### Phase 6.5 — Cron-driven periodic health check (auto-recover loop)

Whenever a RunPod run is launched in Phase 5, set up a periodic remote-status check via cron so a silent failure does not waste GPU budget while the user is away. This is the only polling mechanism this skill uses; the main session itself still must not busy-loop.

Set up the schedule **immediately after Phase 5 succeeds**, in the same turn as the launch report:

1. Use the `schedule` skill (cron-based remote agent) — preferred for runs > ~30 min — or the `loop` skill with a short interval like `/loop 10m` for shorter runs. Recommend `schedule` if the user is going to leave the session.
2. Cadence: **every 10 minutes** is the default. Faster than 5 min wastes context cache; slower than 20 min lets failures sit. Adjust to the run's ETA (longer ETA → 15–20 min cadence is fine).
3. The scheduled job runs `dev/runpod status <run_id>` (and `dev/runpod logs <run_id> --tail 20` if status looks unhealthy) and applies this decision tree:
   - **RUNNING + metrics advancing** → no action, return control.
   - **RUNNING + metrics stalled for ≥2 consecutive checks** → surface to user as a soft warning; do not auto-stop.
   - **EXITED with success marker (`best.pt` + metrics JSON present in S3)** → cancel the cron, jump to Phase 7.
   - **EXITED with failure / crashed / OOM / non-zero exit** → see "Auto-recover loop" below.
   - **Cost cap exceeded** → stop the pod and surface to user immediately. Do NOT auto-relaunch on cost overrun without explicit user approval.
4. Always cancel the cron when Phase 7 starts or when the user terminates the run. Stale cron jobs that keep poking a dead `run_id` are noise.

**Auto-recover loop (failure path).** When the periodic check detects a clear failure (non-zero exit, crashed pod, fatal log line):

1. **Stop** the pod if it's still alive: `dev/runpod stop <run_id>` (confirm with the user only if cost is unclear; for a clearly-crashed pod, stopping is safe).
2. **Pull whatever artifacts and logs exist** so the failure can be diagnosed: `dev/runpod pull <run_id> --from s3` + `dev/runpod logs <run_id>`.
3. **Diagnose the root cause** from the logs. Common patterns: env/setup error (missing dep, dvc pull failure, cwd-relative path — see memory `project_runpod_onstart_pitfalls`), code bug (raised exception in training loop), config mistake (missing `--case`, wrong commit pinned), OOM (batch size too large for the GPU class).
4. **Fix** the root cause in the worktree:
   - Code/config bug → edit, re-run `dev/test-backend`, commit + push to the same branch.
   - OOM → reduce batch size / switch GPU class, commit + push.
   - Onstart / env issue → fix the relevant `dev/runpod` script or case-level setup, commit + push.
   - If the failure is unrelated to the experiment (e.g. RunPod-side outage), surface to user and wait — do NOT auto-relaunch on infra flakes.
5. **Re-launch** with the new commit SHA: `dev/runpod train <new-sha> --case <caseN>`. The `[y/N]` prompt rule from Phase 5 still applies — do not auto-bypass. If the user is not present and the prompt blocks, leave it queued and notify the user.
6. **Re-arm the cron** for the new `run_id` and continue Phase 6.5.

Each step in the auto-recover loop must produce a short Japanese status message to the user (≤ 3 lines) so they can intervene at any point. The loop is **opt-in**: if the user explicitly says they want manual control ("自動で直して回さなくていい", "失敗したら止めるだけでいい"), set up the cron in monitor-only mode (detect & notify, no stop / no fix / no relaunch).

### Phase 7 — Evaluate and document

Once the run finishes:

1. `dev/runpod pull <run_id>` to fetch artifacts to `data/output/models/<family>/case<N>/runs/<run_id>/`.
2. Run the case's local evaluation script (typically `backend/pipeline/<family>/case<N>/evaluation/compare_*.py` or a self-play harness). Use the episode count established in `plan.md` (≥300 if the hypothesis is fragile — n<300 self-play results are not trustworthy on this project per memory `project_imitation_case1_phase3`).
3. **Important**: judge the change on **local match outcomes only**. Do not cite Kaggle publicScore.
4. Write `result.md` (or `iterN_result.md`) in the same directory as `plan.md` with:
   - **Summary** — one paragraph: did the hypothesis hold?
   - **Numbers** — table of win rate / loss / threshold against each opponent, episode counts, run id, commit SHA.
   - **Diagnosis** — why it worked or didn't; any surprising signals.
   - **Decision** — adopted / rejected / needs another iter. If adopted, note whether `dev/runpod promote <run_id>` should be run (and only run it after the user confirms — it overwrites canonical weights).
   - **Artifacts** — paths under `data/output/...` (these are DVC-tracked; do not commit them as files).
5. Report a short summary to the user: hypothesis result, key numbers, recommendation, link to the result file. If the user surfaces a project-level finding worth memoryizing, follow the auto-memory protocol.

Submission to Kaggle (`dev/submit` / `uv run python -m submit submit`) is **out of scope** unless the user explicitly authorizes it within the current request — that consumes the daily 5-submission quota and per `.claude/rules/command.md` requires explicit case+message+mode confirmation immediately before each submission.

For deeper post-mortem (significance testing, replay drill-down, follow-up hypothesis), recommend `/experiment-analysis` rather than expanding `result.md` indefinitely.

## Delegation patterns

The skill runs inline, but you may still delegate narrow sub-tasks via `Task` when it doesn't fragment the user's ability to interject:

- **Heavy planning / architecture for Phase 3**: `Task(subagent_type="planner", ...)` for genuinely structural designs only. Parameter sweeps don't warrant it.
- **Build/type errors during Phase 4**: `python-build-resolver` rather than fighting the toolchain yourself.
- **Code review of the new case after Phase 4**: optional `python-reviewer` pass, worth doing for non-trivial changes (>200 LOC).

Avoid spawning a subagent for the long-running portion (Phase 5–6). That portion must stay in the main session so the user can interject.

## Risk gates this skill enforces

- **RunPod GPU spend is real money.** First launch in a session needs explicit user opt-in.
- **Cron jobs from Phase 6.5 must be cancelled** when Phase 7 starts or when the run is terminated. Stale cron jobs polling dead run_ids are noise and (for `schedule`-based agents) cost.
- **Cost overruns never auto-relaunch.** If the periodic check detects "cost cap exceeded", stop the pod and surface to the user. The auto-recover path applies to crashes/bugs/OOM, not to budget overruns.
- **Kaggle submissions are quota-bound (5/day) and irreversible** — never roll one into an experiment run without immediate explicit user approval per `.claude/rules/command.md`.
- **`dev/runpod promote <run_id>` overwrites canonical weights** in `policy/weights.pt`. Always confirm with the user before running.
- **Self-play results from <300 episodes are noise** on this project (memory `project_imitation_case1_phase3` documents a 5/100 result that did not replicate at 300). When the user asks for a quick decision based on a 100-episode run, surface this caveat.
- **Kaggle publicScore / skill rating must NOT be cited as evidence** for or against a change — opponent pool drift makes them noisy. Local match outcomes only.
- **Never bypass `[y/N]` prompts on `dev/runpod train`.** Scope approval ≠ per-command approval.
- **Don't launch from a dirty tree.** `dev/runpod train` pins from a SHA; uncommitted changes won't make it into the run.

## Common shapes

| User says… | Skill behavior |
|---|---|
| "imitation/case1 で dropout を 0.3 にして回して" | 1 question (確認: RunPod 含めて full run でよいか) → run Phases 2–7 inline. |
| "case4 の plan.md だけ先に書いて" | Skip RunPod question; run Phases 1–3 only, stop after `plan.md`. |
| "さっきの run xxx の結果を docs にまとめて" | Skip Phases 1–5; run Phase 7 (pull + eval + write `result.md`). |
| "rulebase/case2 を新しく作って harass policy 試して" | Confirm new vs existing case → confirm RunPod → run Phases 2–7. |
| "今 RunPod で回ってる run は終わった?" | Phase 6 inline: `dev/runpod status <run_id>`, report concisely. |
| "やっぱり lr も変えたい" (mid-flight) | Treat as a new iter. Do not interrupt the in-flight run. Write `iterN+1_plan.md` and queue a new launch for after the current finishes. |

## Things to avoid

- Spawning a subagent for the long-running portion (Phases 5–6). The user must be able to interject.
- Forgetting to cancel the Phase 6.5 cron when the run finishes or is terminated.
- Auto-relaunching after a cost-overrun stop. Crashes/bugs/OOM are recoverable; budget overruns need explicit user approval.
- Polling more frequently than every 5 minutes. Faster cadence wastes prompt-cache and adds no signal.
- Asking more than 1–2 clarifying questions in Phase 1.
- Launching RunPod without an explicit user opt-in for the first run of the session.
- Polling `dev/runpod status` in a `while` loop. Wait for the user to ask, or use `dev/runpod watch` / `--watch` for one-shot notification.
- Citing Kaggle publicScore as the success metric.
- Creating a new experiment directory when the user is iterating on an existing hypothesis. Use `iterN_*.md` instead.
- Putting machine-generated artifacts (JSON, PNG, replay dumps) under `docs/experiment/`. They go under `data/output/experiment/` and are referenced by path from the markdown.
- Reporting "the model improved" based on Kaggle publicScore — only local match outcomes count.
- Claiming a run finished before pulling and inspecting artifacts.
- Submitting to Kaggle without explicit, in-the-moment user approval.
- Editing `policy/weights.pt` directly. Canonical weights are updated only via `dev/runpod promote <run_id>` after the user confirms.

## Communication cadence

- One short message at each phase boundary (plan written / case implemented / tests green / runpod launched / run finished / result written). Each ≤ 3 lines.
- After RunPod launch, surface the run id and ETA, then return control. Do not narrate while waiting.
- For status checks, ≤ 3 lines: status / latest metric / ETA delta.
- End each turn with what's next ("runpod run started, run_id=abc123, ETA ~45min — `status` で進捗、`止めて` で中断、`結果まとめて` で完了後の集計に進みます").

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and result reports must be in Japanese** (per the project-wide policy in `.claude/CLAUDE.md`)
