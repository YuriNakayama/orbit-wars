---
paths:
  - "pipeline/**"
---

# Pipeline (pipeline/<category>/case*) Submission Rules

Conventions for keeping `pipeline/<category>/case*/` directories runnable **both locally and on Kaggle after submission**. `<category>` currently has two families: `rulebase/` (case0–case2) and `imitation/` (case1). Case numbers are assigned **independently per category** starting from 1 (rulebase/case1 and imitation/case1 are unrelated).

## Premise: submit infrastructure constraints

Constraints imposed by `src/submit/validator.py` and `src/submit/packager.py`:

1. `pipeline/<category>/<case>/main.py` **must exist directly under the case directory** and expose `agent(obs)` at the top level (it is loaded directly via `importlib.util.spec_from_file_location`).
2. The packager bundles `*.py`, `*.json`, `*.yaml`, `*.pkl`, `*.pt`, etc. under `case_dir` into a tar.gz **preserving the relative path structure rooted at `case_dir`**.
3. On the Kaggle runtime, `main.py` is executed directly after the tar.gz is extracted. **The parent package `pipeline.<category>.<case>` does not exist** in the extracted directory, so absolute imports such as `from pipeline.rulebase.case1.xxx import ...` raise `ImportError`.

## Required pattern (Plan B: relative imports + sys.path injection)

### 1) `pipeline/<category>/<case>/main.py` is an entrypoint only

```python
# pipeline/<category>/caseN/main.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
```

- `sys.path.insert(0, str(Path.cwd()))` adds the **runtime cwd** to the top, so `baseline/` becomes importable as a top-level name from the directory where Kaggle extracted the tar.gz.
- `from baseline.agent import agent` is **a path that is not used locally** (locally we use e.g. `pipeline.rulebase.case1.baseline.agent`). This is the key trick that makes both import resolutions succeed from a single source.

### Do not use `__file__` (discovered 2026-04-18)

In the Kaggle sandbox, sys.path injection that depends on `__file__` / `Path(__file__).resolve().parent` causes **Validation Episode failed (`SubmissionStatus.ERROR`)**. The following patterns are **forbidden**:

```python
# NG: causes Validation failed on Kaggle
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

Probable reason: Kaggle's agent loader sets cwd to the extracted directory before launching `main.py`, but `__file__` is environment-dependent (relative paths, symlinks, paths reached via `exec()`, etc.). After `.resolve()` the path may not contain `baseline/`.

**Use `Path.cwd()` instead.** Kaggle is implemented under the assumption that the extracted tar.gz directory is cwd, so `Path.cwd()` reliably points at the extraction directory.

Because of this behavior gap, **the local validator (`src/submit/validator.py`) does not change cwd, so `main.py` written with the `Path.cwd()` pattern fails dry-run with `ModuleNotFoundError`**. To pass `--dry-run` locally, combine it with `--skip-validation`, or `cd pipeline/<category>/case<N>` manually before exec. Other local paths (such as `pipeline.<category>.case<N>.baseline.agent:agent` imported via `src/dataset/selfplay/agents.py`) bypass `main.py` and are therefore unaffected.

### 2) Use relative imports inside subpackages

```python
# pipeline/<category>/caseN/baseline/agent.py
from .core.types import Fleet, Planet           # OK
from .strategy import plan_moves                # OK
from pipeline.rulebase.caseN.baseline.core.types import Fleet, Planet  # NG (unresolvable on Kaggle)

# pipeline/<category>/caseN/baseline/missions/snipe.py
from ..core.config import SNIPE_COST_TURN_WEIGHT  # OK (two package levels up)
from ..strategy_helpers import target_value       # OK
```

- Relative imports **do not depend on the parent package name**, so the same code resolves both locally (via `pipeline.rulebase.caseN.baseline`) and on Kaggle (via `baseline`).
- **Never** write absolute imports of the form `pipeline.<category>.caseN.baseline.*` inside subpackages.

### 3) Subpackage `__init__.py` files also use relative imports

```python
# pipeline/<category>/caseN/baseline/__init__.py
from .agent import agent, build_world

__all__ = ["agent", "build_world"]
```

## Directory layout principles

- `pipeline/<category>/` groups agent families (`rulebase/`, `imitation/`, etc.). When adding a new category, only place an empty `__init__.py`; do not put logic there.
- `pipeline/<category>/case<N>/main.py` is always an entrypoint. Keep it as a thin wrapper of roughly 20 lines. Do not put business logic in it.
- The implementation lives in subpackages under `pipeline/<category>/case<N>/<package>/` (e.g. `baseline/`, `policy/`). Maintain the hierarchy for readability and maintainability.
- Auxiliary directories such as `evaluation/`, `configs/`, `eda/`, `notebook/` may sit under `pipeline/<category>/case<N>/`. They are harmless on Kaggle as long as they are not imported from `main.py`, but the tar.gz size should still be kept small.

## Local-side import paths (unchanged)

The following may continue to be used locally (they remain resolvable after the relative-import refactor):

- `src/dataset/selfplay/agents.py` — `"baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent"`
- `pipeline/<category>/case<N>/evaluation/*.py` — `from pipeline.<category>.case<N>.baseline import agent as baseline_agent`
- `tests/pipeline/<category>/case<N>/*.py` — `from pipeline.<category>.case<N>.baseline.xxx import ...`

These are not bundled into the Kaggle archive, so absolute imports for local-only use are fine.

## Steps to add a new case

1. Create `pipeline/<category>/case<N>/baseline/` (or a strategy-named directory) under an existing category (`rulebase/`, `imitation/`) or a new category, and write all internal imports as relative imports.
2. Create `pipeline/<category>/case<N>/main.py` from the template above.
3. Use relative imports in `pipeline/<category>/case<N>/baseline/__init__.py` as well.
4. Add `"<name>": "pipeline.<category>.case<N>.baseline.agent:agent"` to `AGENT_REGISTRY` in `src/dataset/selfplay/agents.py`.
5. Run `dev/submit <category>/case<N> --dry-run -m "..."` and confirm that the validator loads `main.py` and `env.run([agent, "random"])` succeeds.
6. Confirm `pytest tests/pipeline/<category>/case<N>` passes before submitting to production.

## Submission archive exclusions (`pipeline/.submitignore`)

The packager reads `pipeline/.submitignore` to decide which paths to exclude from the tar.gz. **It applies across all categories and cases** (a single file directly under the pipeline root).

### Location and syntax

- Location: `pipeline/.submitignore` (a single file directly under the pipeline root)
- Syntax: gitignore-compatible subset
  - Lines starting with `#` are comments; blank lines are ignored
  - A trailing `/` denotes a directory (excludes everything beneath it)
  - Otherwise the pattern is matched against paths/filenames via `fnmatch`
- Paths are evaluated **relative to `case_dir`** (e.g. `eda/` matches `pipeline/rulebase/case1/eda/`)

### Standard exclusion list (this repo)

```
# Development tooling (not needed in the production submission)
eda/
notebook/
evaluation/
training/
configs/
```

Local-only development scripts such as `evaluation/snapshot_update.py` may contain absolute imports, which can cause `ImportError` during Kaggle's file walk and trigger Validation Episode failed. When you add a development subdirectory to a case directory, also add it to `.submitignore`.

### Decision criteria for new development directories

| Directory | Submission artifact? | Add to `.submitignore`? |
|---|---|---|
| `baseline/`, `policy_v2/`, etc. (agent body) | Yes | No |
| `eda/`, `notebook/` (exploration) | No | Yes |
| `evaluation/`, `training/` (development scripts) | No | Yes |
| `configs/` (locally referenced settings) | No (move constants into `core/config.py`) | Yes |
| Model weights `.pt` / `.pkl` | Yes | No |

## Anti-patterns

- Inlining logic in `main.py` — makes it hard to host multiple strategies and to unit-test.
- Using `from pipeline.<category>.caseN.xxx import ...` inside subpackages — `ImportError` on Kaggle.
- Placing `sys.path.insert` outside `main.py` — scatters global side effects and makes them untraceable.
- Writing absolute imports in `__init__.py` — defeats the relative-import refactor.
- **Injecting into sys.path via `__file__` — causes Validation Episode failed on Kaggle**. Always use `Path.cwd()` (see the "Do not use `__file__`" section above).

## Submission quota behavior (discovered 2026-04-18)

Kaggle Orbit Wars allows up to 5 submissions per day, but `SubmissionStatus.ERROR` (validation failures) are **not** counted against the quota. In other words, **submissions that fail validation can be retried immediately**: when an error occurs, inspect the Kaggle Web UI logs, identify the cause, and resubmit right away.

`src/submit/` does not perform a local quota check (if the Kaggle side is over the limit, the submit simply fails — the local-side aggregation can have timing skew, so we avoid spurious local rejections). Check the current submission count with `uv run python -m submit submissions`.

## Evaluation metric interpretation (local match results only)

Kaggle Orbit Wars publicScore and skill rating are **relative metrics** computed against other participants' submissions, and the opponent pool drifts over time, so **the same agent can produce very different publicScores depending on submission timing**. Treat Kaggle-side numbers as unreliable for judging the merit of a change. **Evaluate agents exclusively on local match results.** Concretely:

- **Use only local match outcomes** (self-play and head-to-head between local agents) when comparing strategies. Run scripts such as `pipeline/<category>/case<N>/evaluation/compare_*.py` and judge based on win rate over a sufficient number of episodes.
- **Do not use Kaggle publicScore / skill rating as an evaluation signal.** Even when comparing submissions made around the same time, do not infer ranking from publicScore.
- When publicScore happens to be observed (e.g. for monitoring purposes), it is acceptable to record it together with the submission datetime, but it must not be quoted as evidence for or against any change.

## Verification commands

```bash
# Verify the local import path
uv run python -c "from pipeline.rulebase.case1.baseline.agent import agent; print(agent)"

# Simulate the Kaggle-side import path (equivalent to the validator that loads main.py directly)
uv run python -m submit submit rulebase/case1 --dry-run -m "dry-run verification"

# Pre-submission check suite
dev/test-backend
```
