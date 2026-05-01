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
- **File name**: `plan.md` for proposals (before running), `result.md` for outcomes (after running), or another short noun (e.g. `analysis.md`, `notes.md`) when neither fits. A single experiment directory may contain both `plan.md` and `result.md` paired together.
- **Iterations**: when the same experiment goes through multiple iterations that each warrant their own document, do NOT create new directories. Add files within the existing directory using an `iter{N}_` prefix (e.g. `iter1_result.md`, `iter2_plan.md`, `iter2_result.md`). Keep all iterations under the same directory so the experiment's history stays together.
- **Iteration migration**: when introducing the `iter{N}_` scheme into a directory that already has a plain `result.md` / `plan.md`, rename the existing file to `iter1_result.md` / `iter1_plan.md` first, then add `iter2_*.md` for the new work. Don't leave the original unprefixed file alongside `iter2_*.md` — it makes the iteration order ambiguous.

### Examples

```
docs/experiment/imitation/20260419_case1_diagnosis/result.md
docs/experiment/imitation/20260422_case2_features/plan.md
docs/experiment/rulebase/20260418_case1_optimize/plan.md
docs/experiment/imitation/20260420_case1_pure_il_cycle/iter1_result.md   # iteration variant
docs/experiment/imitation/20260420_case1_pure_il_cycle/iter2_result.md
```

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
