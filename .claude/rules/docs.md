---
paths:
  - "docs/**"
---

# Documentation Rules

Conventions for the `docs/` tree. Keep human-readable narrative under `docs/`; put machine-generated artifacts (JSON metrics, PNG plots, replay dumps) under `data/output/` so docs stay diff-friendly and lightweight.

## Top-level layout

```
docs/
  competition/    Competition spec & rules summaries (abstract.md, baseline / evaluation references)
  experiment/     Experiment plans and results (see structure below)
  plans/          Feature plans (one directory per feature, numbered phases 00-codebase-research → 06-testing)
```

`docs/competition/` holds reference material about the competition itself — rules, scoring, baseline mechanics. It does not change with each experiment iteration.

## `docs/experiment/` structure

Use this exact path scheme:

```
docs/experiment/{imitation|rulebase|reinforce}/{yyyymmdd}_case{N}_{topic}/{plan|result|etc.}.md
```

- **First segment**: agent family the experiment targets — `imitation`, `rulebase`, or `reinforce`.
- **Directory name**: `{yyyymmdd}_case{N}_{topic}` — the date the experiment started, the case number it operates on, and a short snake_case topic.
- **File name**: `plan.md` for proposals (before running), `result.md` for outcomes (after running), `hypotheses.md` for the case-level hypothesis list + skip list (single source of truth for the `experiment` loop driver — created by `experiment-hypothesize`, updated by `experiment-execution` / `experiment-analysis`), or another short noun (e.g. `analysis.md`, `notes.md`) when none fit. A single experiment directory may contain `hypotheses.md` together with paired `plan.md` / `result.md` (or their `iter{N}_*` variants).
- **Iterations**: when the same experiment goes through multiple iterations that each warrant their own document, do NOT create new directories. Add files within the existing directory using an `iter{N}_` prefix (e.g. `iter1_result.md`, `iter2_plan.md`, `iter2_result.md`). `hypotheses.md` is **not** iter-prefixed — there is one per directory, and iterations are tracked inside it via the `Iteration log` section + checkbox state on each hypothesis. Keep all iterations under the same directory so the experiment's history stays together.
- **Iteration migration**: when introducing the `iter{N}_` scheme into a directory that already has a plain `result.md` / `plan.md`, rename the existing file to `iter1_result.md` / `iter1_plan.md` first, then add `iter2_*.md` for the new work. Don't leave the original unprefixed file alongside `iter2_*.md` — it makes the iteration order ambiguous.

### Examples

```
docs/experiment/imitation/20260419_case1_diagnosis/result.md
docs/experiment/imitation/20260422_case2_features/plan.md
docs/experiment/rulebase/20260418_case1_optimize/plan.md
docs/experiment/imitation/20260420_case1_pure_il_cycle/iter1_result.md   # iteration variant
docs/experiment/imitation/20260420_case1_pure_il_cycle/iter2_result.md
docs/experiment/imitation/20260420_case1_pure_il_cycle/hypotheses.md     # case-level hypothesis list (one per dir, no iter prefix)
```

### `hypotheses.md` (case-level hypothesis list)

`hypotheses.md` is the **single source of truth** for the `experiment` loop driver. It captures the hypothesis list to be tested, the priority order, and — critically — the **skip list** (`実施しない検証 / 評価`) that downstream skills (`experiment-plan` / `experiment-execution` / `experiment-analysis`) honor to short-circuit their own phases.

Required sections:

- **Header frontmatter** (Markdown blockquote): created date / last updated / state (`in_progress` / `paused` / `stopped` / `completed`) / max iterations / primary metrics / default episode count.
- **`## 実施しない検証 / 評価 (skip list)`** — must always be present. Subsections:
  - `### 評価` (Evaluation) — e.g. `do not run local self-play 300 matches (adopt/reject on loss curve only)`, `do not quote Kaggle publicScore`, `decide adoption on 100 matches only`
  - `### 分析` (Analysis) — e.g. `do not run replay analysis`, `do not draw conclusions from n<300 results`
  - `### 実行` (Execution) — e.g. `skip smoke test (1-episode self-play)` ⚠️, `skip dev/test-backend` ⚠️, `do not use RunPod GPU (local CPU only)`, `do not use the auto-recover loop`
  - `### 例外条件` (Exception conditions) — per-hypothesis overrides (e.g. `add 300 matches for H4 only when inconclusive`)
- **`## 仮説リスト (priority 順)`** (Hypothesis list, in priority order) — checkbox + priority label + (optional) `depends on H{m}`:
  ```markdown
  - [ ] (P1) H1: dropout 0.2→0.3 — expect win-rate +3pp from overfitting suppression
  - [x] (P2) H3: focal α 0.25→0.75 — adopted (iter9)
  - [ ] (P2, depends on H1) H4: dropout=0.3 + add in-flight ratio to features
  - [ ] (deferred) H5: optimizer Adam→Lion — high effort
  ```
- **`## Iteration log`** — table (iter / start / hypothesis# / plan path / run_id / primary metrics / adopt-reject / result path / analysis path). `experiment-execution` Phase 8 appends a row; `experiment-analysis` Phase 4.5 updates the checkbox + adoption note.

Project-default skip-list entries (always include unless the user explicitly removes them):

- `Kaggle publicScore は引用しない` (memory `project_om_finding`, `project_case5_validation`)
- `n<300 結果で結論を出さない` (memory `project_imitation_case1_phase3`)

Lifecycle:

- Created **once** per case directory by `experiment-hypothesize`.
- Updated by `experiment-execution` (Iteration log row append) and `experiment-analysis` (checkbox + adoption note + state-transition flag).
- Read by `experiment-plan` (skip list → auto-shortens the plan's `## 検証方法` section) and `experiment` (loop driver — picks the next `- [ ]` hypothesis).

> **Note:** The Japanese strings quoted in backticks above (`## 実施しない検証 / 評価`, `## 仮説リスト`, `### 評価` / `### 分析` / `### 実行` / `### 例外条件`, `## 検証方法`, and the default skip-list entries `Kaggle publicScore は引用しない` / `n<300 結果で結論を出さない`) are **literal contract strings** that the `experiment-*` skills emit and pattern-match against. They are part of the `hypotheses.md` schema, not prose — do not translate them, or the skill cross-references will break.
- Not iter-prefixed. One per case directory.

### Anti-patterns

- Don't put dates in file names (`20260419_diagnosis_result.md`). The date belongs to the directory.
- Don't omit `case{N}` — every experiment is scoped to a specific case.
- Don't drop non-Markdown artifacts (`*.json`, `*.png`, replay dumps) into `docs/experiment/`. Write them under `data/output/experiment/` and link from the Markdown if needed.
- Don't mix experiment results into `docs/competition/`. Competition material is reference; experiments are work logs.

## Cross-references

- Use repo-rooted paths (e.g. `docs/experiment/imitation/20260419_case1_diagnosis/result.md`) when referencing across categories — they survive future reorgs better than relative paths.
- Within the same category subtree, relative paths (`../20260420_case1_improve/result.md`) are fine.
- When referencing artifacts under `data/output/`, write the path as-is (`data/output/experiment/imitation_case1_val_metrics.json`); these files are not git-tracked but are reproducible via DVC.

## When to add a new experiment directory

A new directory is warranted when an experiment has a distinct hypothesis, dataset, or evaluation. Iterations of the same hypothesis stay in the same directory: either appended into the existing `result.md`, or split across `iter{N}_result.md` files when each iteration is large enough to warrant its own document.
