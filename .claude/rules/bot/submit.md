---
paths:
  - "bot/src/submit/**"
  - "dev/submit"
---

# Pipeline Submit Rules

Conventions for keeping `bot/pipeline/<category>/case*/` directories runnable **on Kaggle after submission**. These rules are loaded when editing the submit implementation (`bot/src/submit/**`) or the `dev/submit` wrapper. For directory layout, training execution, and evaluation conventions that apply to all pipeline work, see [`pipeline.md`](pipeline.md).

Path notation below uses `bot/` as the anchor (`pipeline/<category>/case<N>/...`). `uv run ...` / `dev/submit ...` are expected to execute with `bot/` as the working directory.

## Premise: submit infrastructure constraints

Constraints imposed by `bot/src/submit/validator.py` and `bot/src/submit/packager.py`:

1. `pipeline/<category>/<case>/main.py` **must exist directly under the case directory** and expose `agent(obs)` at the top level (it is loaded directly via `importlib.util.spec_from_file_location`).
2. The packager bundles `*.py`, `*.json`, `*.yaml`, `*.pkl`, `*.pt`, etc. under `case_dir` into a tar.gz **preserving the relative path structure rooted at `case_dir`**.
3. On the Kaggle runtime, `main.py` is executed directly after the tar.gz is extracted. **The parent package `pipeline.<category>.<case>` does not exist** in the extracted directory, so absolute imports such as `from pipeline.rulebase.case1.xxx import ...` raise `ImportError`.

## Required pattern (Plan B: relative imports + sys.path injection)

### 1) `pipeline/<category>/<case>/main.py` is an entrypoint only

```python
# pipeline/<category>/caseN/main.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
```

- `sys.path.insert(0, str(Path.cwd()))` adds the **runtime cwd** to the top, so `baseline/` becomes importable as a top-level name from the directory where Kaggle extracted the tar.gz.
- `from baseline.agent import agent` is **a path that is not used locally** (locally we use e.g. `pipeline.rulebase.case1.baseline.agent`). This is the key trick that makes both import resolutions succeed from a single source.

- On Kaggle, sys.path injection that depends on `__file__` causes **Validation Episode failed**. The following pattern is **forbidden**:

```python
# NG: causes Validation failed on Kaggle
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

**Use `Path.cwd()` instead.** Kaggle is implemented under the assumption that the extracted tar.gz directory is cwd.

Because of this behavior gap, **the local validator (`bot/src/submit/validator.py`) does not change cwd, so `main.py` written with the `Path.cwd()` pattern fails dry-run with `ModuleNotFoundError`**. To pass `--dry-run` locally, combine it with `--skip-validation`, or `cd bot/pipeline/<category>/case<N>` manually before exec.

### 2) Use relative imports inside subpackages

```python
# pipeline/<category>/caseN/baseline/agent.py
from .strategy import plan_moves                # OK
from pipeline.rulebase.caseN.baseline.core.types import Fleet, Planet  # NG (unresolvable on Kaggle)

# pipeline/<category>/caseN/baseline/missions/snipe.py
from ..core.config import SNIPE_COST_TURN_WEIGHT  # OK (two package levels up)
```

- Relative imports **do not depend on the parent package name**, so the same code resolves both locally (via `pipeline.rulebase.caseN.baseline`) and on Kaggle (via `baseline`).
- **Never** write absolute imports of the form `pipeline.<category>.caseN.baseline.*` inside subpackages.

### 3) Subpackage `__init__.py` files also use relative imports

```python
# pipeline/<category>/caseN/baseline/__init__.py
from .agent import agent, build_world

__all__ = ["agent", "build_world"]
```

## Submission archive exclusions (`bot/pipeline/.submitignore`)

The packager reads `bot/pipeline/.submitignore` to decide which paths to exclude from the tar.gz. **It applies across all categories and cases** (a single file directly under the pipeline root).

- Location: `bot/pipeline/.submitignore` (a single file directly under the pipeline root)
- Syntax: gitignore-compatible subset (matched via `fnmatch`)
- Paths are evaluated **relative to `case_dir`** (e.g. `eda/` matches `bot/pipeline/rulebase/case1/eda/`)

## Submit verification steps

When adding or changing a submittable case, confirm the Kaggle import path before a real submission:

1. Ensure `pipeline/<category>/case<N>/main.py` follows the entrypoint template above, and all internal imports (including `__init__.py`) are relative.
2. Add `"<name>": "pipeline.<category>.case<N>.baseline.agent:agent"` to `AGENT_REGISTRY` in `src/dataset/selfplay/agents.py` (local-side only; not bundled into the tar).
3. Run `dev/submit <category>/case<N> --dry-run -m "..."` and confirm the validator loads `main.py` and `env.run([agent, "random"])` succeeds.
4. Confirm `pytest tests/pipeline/<category>/case<N>` passes before submitting to production.

## Anti-patterns (submit)

- Inlining logic in `main.py` — makes it hard to host multiple strategies and to unit-test.
- Using `from pipeline.<category>.caseN.xxx import ...` inside subpackages — `ImportError` on Kaggle.
- **Injecting into sys.path via `__file__`** — causes Validation Episode failed on Kaggle. Always use `Path.cwd()`.
