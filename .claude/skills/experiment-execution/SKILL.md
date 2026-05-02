---
name: experiment-execution
description: >
  Execution runner for Orbit Wars experiments under `backend/pipeline/`. Drives
  the `experimenter` subagent through the full cycle: capture the hypothesis →
  write `docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/plan.md` → implement
  / extend the case under `backend/pipeline/{rulebase|imitation|reinforce}/case<N>/`
  → run `dev/test-backend` for local bug detection → push the commit and launch
  GPU training on Vast.ai via `dev/vast train` → report progress periodically →
  evaluate on local match outcomes and write `result.md`. Use whenever the user
  types `/experiment-execution`, or asks to run / execute / iterate / kick off an
  experiment, train a new model, launch a Vast.ai run, propose a new case, or
  write up an experiment result — even if they don't explicitly say "execute",
  phrases like "imitation/case1 で dropout を試したい",
  "rulebase/case2 を改良して回したい", "vast で学習を回して結果まとめて",
  "新しい case を切って学習させたい", "plan.md に従って実装して回して" all
  count. Don't trigger this skill for hypothesis-only discussion / planning
  conversations (use `experiment-plan`), interactive result interpretation /
  follow-up analysis after a run finishes (use `experiment-analysis`),
  read-only code review, plain bug fixes, or Kaggle submission requests.
---

# Experiment Execution Skill (Orbit Wars)

This skill is the front door for any experiment that lives under `backend/pipeline/`. Its job is to translate a user's intent ("imitation/case1 で dropout を上げて回したい") into a clean run of the **`experimenter` subagent**, which owns the heavy lifting (plan.md, case implementation, pytest, Vast.ai launch, status reporting, result.md).

The skill itself is thin on purpose. It captures intent, gates risky actions, and hands off to the agent. The agent is where the workflow lives — see `.claude/agents/experimenter.md`.

## When this skill is in charge

- The user typed `/experiment` (explicit trigger), **or**
- The user asked, in plain language, for any of:
  - a new experiment / case (`新しい case を切って…`, `RL の baseline を回したい`)
  - an iteration of an existing case (`imitation/case1 を改良して再学習`, `case4 の harass policy を強くして…`)
  - a Vast.ai training run (`vast で学習回して`, `GPU で学習させたい`)
  - writing up the result of a recent run (`さっきの run 結果を docs にまとめて`)
- The user did **not** already pre-specify the entire pipeline themselves (e.g. "just run `dev/vast train abc123`" — at that point, just run it).

If the user is asking for **read-only review**, **dead-code cleanup**, **Kaggle submission**, or **infra-only work**, redirect to the matching skill/agent (`code-review`, `python-review`, the submission flow under `dev/submit`, etc.) instead. Experiments are about hypothesis → trained model → result, not about polishing existing code.

## What "an experiment" means in this repo

A single experiment is bounded by:

- **Family** — `imitation` / `rulebase` / `reinforce`. Determined by which agent type the change targets.
- **Case** — `case<N>` under `backend/pipeline/<family>/`. New case vs. extending an existing one is a real fork: extending preserves comparable baselines, new cases are warranted only when the hypothesis genuinely doesn't fit the existing structure.
- **Hypothesis** — one sentence on the change being tested + the success metric (typically a local-match win rate against a named opponent, ≥300 episodes for fragile claims).
- **Compute target** — local CPU only, or Vast.ai GPU. Anything beyond a few minutes of training goes on Vast.

Read `.claude/rules/docs.md` for `docs/experiment/...` naming and `.claude/rules/backend/pipeline.md` for case-directory submit constraints. The `experimenter` agent already encodes both, so the skill doesn't restate them — but if you (the assistant in the main session) are asked to do anything yourself outside of delegating, follow those rules.

## Skill flow

The skill itself runs four short phases. Phase 3 is the only one that does meaningful work — everything else is making sure the handoff to the agent is clean.

### Phase 1 — Confirm scope before delegating

Before spawning the agent, the user should know what they're authorizing. In particular, **GPU training on Vast.ai costs real money**, so don't sleepwalk into a launch.

Use `AskUserQuestion` (Japanese labels — see Language section) to lock down anything still ambiguous in the request. Aim for **at most 1–2 questions**; skip any field already implied by the user. Likely fields:

- **Family + case**: which family, new case (next free `caseN`) or extend an existing case?
- **Compute target**: local pytest only, or include a Vast.ai run? If Vast, present a brief one-line cost expectation so the user opts in knowingly.
- **Scope of this skill invocation**: full pipeline (plan → impl → test → vast → result), partial (e.g. plan.md only, or "just write up the result of run xyz"), or "dry-run plan only".

If the user already said `dry-run のプランだけ` or `plan.md だけ書いて`, do not ask about Vast — they've already scoped it down.

### Phase 2 — Quick reality check

Before spawning the agent, run these in parallel via Bash. They're cheap and surface blockers early:

```bash
git status --short          # is the working tree clean enough to push?
git rev-parse --abbrev-ref HEAD   # which branch will dev/vast train pin from?
ls backend/pipeline/<family>/      # which case numbers already exist?
```

If `git status` is dirty and the user wants a Vast launch, surface that to the user and decide together: commit first, stash, or proceed without Vast. Do NOT silently launch from a dirty tree.

### Phase 3 — Hand off to the `experimenter` agent

Spawn the agent via the `Task` tool with `subagent_type: "experimenter"`. The prompt should be self-contained: the agent does not see this conversation.

What to include in the agent prompt:

1. **The user's request, verbatim** (or a faithful Japanese paraphrase if the user spoke obliquely).
2. **Scope decisions from Phase 1** — family, case (new or existing N), compute target, dry-run vs full run.
3. **Branch + commit context** — current branch name, current SHA, "tree is clean / dirty" from Phase 2.
4. **Explicit non-goals** — e.g. "do not run `dev/submit submit` to Kaggle", "do not promote weights via `dev/vast promote`", "only write `plan.md`, do not implement code". Be specific; the agent is conservative but not psychic.
5. **What to return** — a short report covering: where `plan.md` was written, what was implemented, test status, vast run id (if launched), ETA, next-step recommendation.

Example handoff prompt:

```
User wants to run an experiment iterating on imitation/case1.

Hypothesis (from user): "dropout を 0.2 → 0.3 に上げて再学習。LB1235 baseline 相手の局所 win-rate 改善を狙う。"

Scope confirmed in main session:
- Family: imitation, case: existing case1 (extend, not new case)
- Compute: full pipeline including Vast.ai GPU training
- This is iter 4 of an existing experiment dir (check
  docs/experiment/imitation/ for the matching directory and use
  iterN_plan.md / iterN_result.md naming if applicable)

Branch context: feature/reinforcement-learning-conversion @ <SHA>, tree clean.

Non-goals:
- Do NOT submit to Kaggle.
- Do NOT promote weights (dev/vast promote) without explicit follow-up.

Please execute Phases 2–6 of your workflow and return:
- plan.md path
- pytest result (pass/fail summary)
- vast run_id, stage, ETA
- A one-paragraph next-step recommendation
```

After spawning, the agent runs autonomously. Surface its summary to the user verbatim (translated to Japanese if needed); do not re-derive the work yourself.

### Phase 4 — Follow-up coordination

After the agent returns, two things commonly need follow-up; handle them in the main session rather than re-spawning the agent:

- **Vast.ai status checks while the run is in flight.** When the user asks for an update later, you can either re-spawn `experimenter` with the run id, or run `dev/vast pull <run_id>` directly and report metrics. Re-spawning is preferable if the user also wants a written `result.md` update.
- **Promotion / submission asks.** If after `result.md` the user wants `dev/vast promote <run_id>` or a Kaggle submission, treat each as its own approval gate per `.claude/rules/command.md` — these are irreversible and need explicit confirmation immediately before each action.

## Risk gates this skill enforces

Even though the agent has its own guardrails, the skill is the user-visible layer and must not let these slip:

- **Vast.ai GPU spend is real money.** First launch in a session needs explicit user opt-in. Subsequent launches in the same conversation under the same approved cost envelope do not need re-confirmation.
- **Kaggle submissions are quota-bound (5/day) and irreversible** — never roll one into an experiment run without immediate explicit user approval per `.claude/rules/command.md`. The skill does not submit to Kaggle as part of the experiment loop.
- **`dev/vast promote <run_id>` overwrites canonical weights** in `policy/weights.pt`. Always confirm with the user before running.
- **Self-play results from <300 episodes are noise** on this project (memory `project_imitation_case1_phase3` documents a 5/100 result that did not replicate at 300). When the user asks for a quick decision based on a 100-episode run, surface this caveat.
- **Kaggle publicScore / skill rating must NOT be cited as evidence** for or against a change — opponent pool drift makes them noisy. Local match outcomes only.

## Common shapes

| User says… | Skill behavior |
|---|---|
| "imitation/case1 で dropout を 0.3 にして回して" | 1 question (確認: Vast 含めて full run でよいか) → spawn agent for full pipeline. |
| "case4 の plan.md だけ先に書いて" | Skip Vast question; spawn agent with `dry-run / plan.md only` non-goal. |
| "さっきの run xxx の結果を docs にまとめて" | Skip Phases 1–2; spawn agent in eval/report-only mode with the run id. |
| "rulebase/case2 を新しく作って harass policy 試して" | Confirm new vs existing case → confirm Vast → spawn full pipeline. |
| "今 Vast で回ってる run は終わった?" | No agent needed; run `dev/vast pull <run_id>` directly and report. (If the user also wants result.md, then spawn agent.) |

## Things to avoid

- Re-implementing what the agent does (writing `plan.md` yourself instead of delegating). The agent is the implementation; the skill is just the gate.
- Asking more than 1–2 clarifying questions. The agent will ask for missing pieces itself if needed.
- Launching Vast without an explicit user opt-in for the first run of the session.
- Citing Kaggle publicScore as the success metric.
- Creating a new experiment directory when the user is iterating on an existing hypothesis. The agent already enforces `iterN_*.md` naming under the existing directory; trust it.

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and result reports must be in Japanese** (per the project-wide policy in `.claude/CLAUDE.md`)
