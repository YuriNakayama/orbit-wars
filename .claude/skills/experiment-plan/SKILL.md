---
name: experiment-plan
description: >
  Interactive planning skill for Orbit Wars experiments under `backend/pipeline/`.
  Captures the experiment overview from the user (or from `/experiment-plan <args>`),
  surveys existing case code under `backend/pipeline/<family>/case<N>/` and prior
  experiment docs, walks the user through hypothesis + validation method via
  `AskUserQuestion` (selection-based with free-text via "Other"), optionally
  performs web research when the user requests it, and writes a single
  `docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/{plan.md | iterN_plan.md}`.
  Stops at the plan; does NOT implement code, push commits, or launch GPU
  training. Use whenever the user types `/experiment-plan`, or asks to plan /
  design / discuss / write up a new experiment hypothesis — phrases like
  "imitation/case1 で dropout 上げる実験の plan を書きたい",
  "新しい case を切る前に仮説を整理したい",
  "rulebase/case2 改良の plan.md だけ先に書いて", "実験設計を相談したい",
  "次の iter の plan を作って" all count. Don't trigger this skill for
  full-pipeline execution including RunPod training (use `experiment-execution`),
  result interpretation after a run finishes (use `experiment-analysis`),
  large-scale multi-domain feature planning (use `feature-plan`), read-only
  code review, or plain bug fixes.
---

# Experiment Plan Skill (Orbit Wars)

Interactive front door for **planning** an Orbit Wars experiment without executing it. The skill captures the experiment overview, investigates the existing case code, runs a structured hearing on hypothesis + validation method, optionally pulls in web research, and writes a single `plan.md` (or `iterN_plan.md`) under `docs/experiment/`. After this skill finishes, the user can review, edit, or hand off to `experiment-execution`.

## When this skill is in charge

- The user typed `/experiment-plan` (explicit trigger), with or without an inline overview argument, **or**
- The user asked, in plain language, for any of:
  - drafting a `plan.md` for a new or existing case
  - structuring an experiment hypothesis before committing to code
  - turning a vague idea into a concrete experiment scope
- The user did **not** ask to also implement / train / push. If they did, redirect to `experiment-execution`.

If the user asks to **execute** an experiment end-to-end, **interpret a finished run**, **survey papers as a primary task**, or **plan multi-domain features**, use the matching skill (`experiment-execution` / `experiment-analysis` / `research-*` / `feature-plan`) instead.

## Scope boundaries (what this skill does NOT do)

- Does not run `dev/test-backend`, `dev/runpod train`, or any other build / training command.
- Does not edit code under `backend/pipeline/<family>/case<N>/`. The plan refers to files by path; it doesn't create them.
- Does not commit or push.
- Writes `plan.md` directly in the main session (no subagent handoff). Hand off to `experiment-execution` only when the user is ready to implement and train.

## Skill flow

The skill runs five steps. Step 0 is conditional, Steps 1 and 2 are the core, Step 3 is opt-in, Step 4 writes and reports.

### Step 0 — Capture the overview (conditional)

If the user invoked `/experiment-plan` with an inline overview argument, or if the natural-language request already contains a clear overview ("imitation/case1 で dropout を 0.2→0.3 に上げる plan を書いて"), **skip this step**.

Otherwise, ask the user for a free-text overview via `AskUserQuestion` with a single open-ended question. Keep it minimal — one question, "Other" free-text expected to carry the answer. Suggested options for the dropdown can include 2–3 plausible templates ("imitation の既存 case に対する hyperparameter 変更", "rulebase の新しい戦術追加", "新しい case を切る") to seed the user's thinking.

After this step you should know roughly: agent family, target case (existing or new), and one-sentence intent. Anything still ambiguous (e.g. "rulebase の何 case?") gets resolved in Step 1's read-back.

### Step 1 — Investigate existing case code

Before any hearing, read the relevant code so the next step's questions are specific. This step runs without user interaction.

Mandatory reads (skip those that don't apply):

- `ls backend/pipeline/<family>/` — confirm existing case numbers; identify the next free `case{N+1}` slot if the user wants a new case.
- For an **existing-case experiment**: read `backend/pipeline/<family>/case<N>/` — at minimum `main.py`, `agent.py`, the primary policy/strategy module, and any module the user named in the overview. Trace what currently exists so the plan can describe the diff precisely (file paths, function names).
- `ls docs/experiment/<family>/` — surface prior experiment directories. If a directory matches the same hypothesis (same case + same topic family), this is an **iteration**, not a new experiment — read its latest `result.md` / `analysis.md` so the plan can reference it (`関連:` line) and avoid repeating mistakes.
- `backend/src/dataset/selfplay/agents.py` (only if registering a new case is implied) — note what's in `AGENT_REGISTRY` so the plan's "Implementation outline" can include the registration step.

Keep the reads narrow. The goal is enough context to ask Step 2's questions specifically — not a full code review.

After investigating, surface a 3–5 line summary back to the user (in Japanese): "対象は `pipeline/<family>/case<N>/`、現状は X / Y / Z で構成、関連する過去 iter は ...". This is the orientation step before the hearing.

### Step 2 — Hearing: hypothesis & validation method

Run **2 rounds** of `AskUserQuestion` (each round up to 4 questions). All questions use the selection format with free-text via "Other"; the first question of the first round explicitly mentions: `各質問で「Other」を選ぶと自由記述も可能です。`

When a question has real trade-offs, use the comparison-style options (`⭐推薦: option — ✅ pro / ⚠️ con` followed by `💡 推薦理由`). Plain options for factual questions.

**Round 1: Hypothesis & success criteria**

Pick the 3–4 most relevant; skip what's already in the overview from Step 0:

- **Hypothesis (one sentence)**: free-text via "Other" is the common path; offer 2–3 plausible phrasings drawn from Step 1's findings to seed the user.
- **Mechanism / why-it-should-work**: brief rationale. Helps catch fuzzy hypotheses early.
- **Primary metric**: local self-play win rate vs. named opponent (most common) / loss curve / specific behavior. **Kaggle publicScore is NOT a valid metric** and must not be offered as an option (project-wide rule per memory `project_om_finding`, `project_case5_validation`).
- **Opponent pool**: which baseline(s) to play against. List what's actually registered in `AGENT_REGISTRY` from Step 1's read.
- **Episode count & decision threshold**: pre-fill `⭐推薦: ≥300 episodes` per memory `project_imitation_case1_phase3` (n<300 not trustworthy on this project).

**Round 2: Validation method (scope, validation, compute)**

Pick the 3–4 most relevant:

- **Scope**: which files/modules under `backend/pipeline/<family>/case<N>/` change. For existing-case extension, recommend **adding a new module** (`policy_v2.py`, `features_v3.py`) over rewriting in place — preserves a baseline. Surface as comparison-style choice.
- **Hyperparameter / config changes**: what knobs move, from what to what.
- **Local validation**: which `pytest` paths cover the change (`tests/pipeline/<family>/case<N>/`). Always recommend `dev/test-backend` before remote launch.
- **Compute target**: local CPU only / RunPod GPU. Pre-fill `⭐推薦: RunPod` for model training >few minutes; pre-fill `⭐推薦: local only` for rule-based or evaluation-only changes.
- **RunPod case identifier** (only if RunPod was selected): existing case like `--case case1` vs. a new case requiring scaffolding under `backend/pipeline/<family>/case<N>/`.

If the user pushes back on a pre-filled `⭐推薦`, follow their lead — recommendations are defaults, not gates.

### Step 3 — Web research (opt-in)

After Step 2, ask the user a single question via `AskUserQuestion`:

> "外部 web 調査が必要ですか? (例: 類似の Kaggle 解法、論文の手法、library のドキュメント確認)"

Options:

- `不要 — Step 1 のコード調査で十分`
- `必要 — 具体的なクエリは Other で指定`
- `必要 — トピックを skill 側で提案`

If the user picks "不要", skip to Step 4.

If "必要", run `WebSearch` / `WebFetch` for 2–4 queries scoped to the user's specification. Bias toward authoritative sources: official docs, Kaggle writeups, arXiv, well-cited GitHub repos. Capture key findings (1–3 bullet points per source with URL) — these will be folded into the plan's "References" section. Time-box the research: aim for ≤ 5 minutes of tool calls; if a topic warrants deeper survey, recommend `/research-retrieval` and stop.

### Step 4 — Write `plan.md`

Resolve the target path:

```
docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/{plan.md | iterN_plan.md}
```

- `{yyyymmdd}` — today's date from the session's `currentDate` (NOT shell `date`, which can drift in sandboxes).
- `{topic}` — short snake_case noun phrase from the hypothesis (`dropout_sweep`, `harass_policy`, `pure_il_cycle`).
- **Iteration handling** per `.claude/rules/docs.md`:
  - Same hypothesis as an existing directory → add `iterN_plan.md` inside it (do NOT create a new directory).
  - First time introducing the `iterN_*` scheme into a directory that has plain `plan.md` → rename the existing file to `iter1_plan.md` first, then write `iter2_plan.md`. Surface the rename to the user before doing it.

Write the plan with these sections (keep it to roughly one screen — verbose plans rot):

```markdown
# {Family}/{caseN} — {Topic}

> 作成日: {yyyy-mm-dd}
> 関連: {paths to prior plan.md / result.md / analysis.md if iterating}
> スコープ: {one-line scope}

## 仮説 (Hypothesis)
{one-sentence hypothesis} — {why it should work}

## 既存コードの現状 (from Step 1)
- 主要モジュール: `backend/pipeline/<family>/case<N>/...` の {要点}
- 過去 iter の所見: {related result.md/analysis.md からの 1-2 行}

## スコープ (Scope)
- 変更ファイル: `backend/pipeline/<family>/case<N>/...`
- ハイパーパラメータ / config: {before → after}
- データセット / 特徴量変更: {if any, else 「なし」}

## 実装ステップ (Implementation outline)
1. {step with file path}
2. ...

## 検証方法 (Validation method)
- ローカル: `dev/test-backend` + `uv run --directory backend pytest tests/pipeline/<family>/case<N> -x`
- (submit-shape change の場合) `uv run --directory backend python -m submit submit <family>/case<N> --dry-run`
- リモート: `dev/runpod train --case {caseN or 「RunPod 不要」}`、想定所要時間 {if known}
- 評価: 対戦相手 {opponents}、エピソード数 {N} ({rationale})、主要メトリクス {metric}、採否しきい値 {threshold}

## 参考 (References) — Step 3 を実施した場合のみ
- {URL} — {1 行所感}
```

Write directly with `Write` (or `Edit` for renames) in the main session.

### Step 5 — Report and offer next steps

After writing the file, report to the user (in Japanese):

- Path of the written `plan.md`
- A 2–3 line summary of the locked decisions (hypothesis, primary metric, compute target)
- One-line offer of next steps:
  - `/experiment-execution` でこの plan に従って実装 → ローカル検証 → RunPod 学習を回せます
  - もしくは plan.md を直接編集して再レビューも可能です

Do **not** auto-spawn `experiment-execution`. The user explicitly came in for planning; they'll trigger execution themselves if/when ready.

## Risk gates this skill enforces

- **Never offer Kaggle publicScore as a metric.** Opponent pool drift makes it noise. Local match outcomes only.
- **Pre-fill ≥300 episodes for fragile claims.** n<300 self-play is not trustworthy for adoption decisions on this project.
- **Iteration directory hygiene.** When the same hypothesis already has a directory, the plan goes in as `iterN_plan.md`. Gently push back if the user asks for a new directory; comply if they insist.
- **Don't run any execution-side command.** No `dev/test-backend`, no `dev/runpod train`, no `uv run pytest`, no `git commit`. The skill is read-only on code; it only writes the plan (and the `Step 3` web research is read-only).
- **Time-box web research.** If Step 3 takes more than ~5 minutes of tool calls, recommend `/research-retrieval` for the deep dive and stop.

## Common shapes

| User says… | Skill behavior |
|---|---|
| `/experiment-plan imitation/case1 で dropout を 0.3 に上げる` | Step 0 skipped (overview given). Step 1 reads `pipeline/imitation/case1/`. Step 2 asks about metric / opponent / episode count / scope (most fields pre-fillable). Step 3 likely "不要". Step 4 writes `iterN_plan.md` under matching existing dir. |
| "実験の plan を書きたい" (vague) | Step 0 asks for overview. Then Steps 1–4 normally. |
| "case4 の harass policy 改良の plan、Kaggle の Lux S3 の harass 戦術も参考にして書いて" | Step 0 skipped. Step 1 reads `pipeline/rulebase/case4/`. Step 2 hearing. Step 3 explicitly runs because user named external research. Step 4 writes plan with `## 参考` section. |
| "case10 の iter5 の plan を作って" | Step 0 skipped. Step 1 finds matching existing directory + reads prior `iter4_result.md`. Step 2 hearing references prior iter findings. Step 4 writes `iter5_plan.md` (or rename + iter5 if needed). |

## Things to avoid

- Asking more than 5 questions across all steps (excluding Step 0's optional overview question and Step 3's opt-in question). The skill is a guided interview, not a survey.
- Forcing comparison-style options on factual questions (e.g. "what's the hypothesis?" doesn't have ⭐推薦).
- Writing implementation code into `plan.md`. The plan refers to file paths; the code itself comes in `experiment-execution`.
- Creating a new experiment directory when the user is iterating on an existing hypothesis.
- Including Kaggle publicScore as a success metric.
- Auto-running `experiment-execution` after writing the plan.
- Letting Step 3 web research balloon into a multi-source survey — that's `/research-retrieval`'s job.

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and the written `plan.md` body must be in Japanese** (per the project-wide policy in `.claude/CLAUDE.md`)
