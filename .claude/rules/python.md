---
paths:
  - "**/*.py"
  - "**/*.ipynb"
---

# Python Rules

**General Python rules** for editing `.py` / `.ipynb` files in this repository. Auto-loaded across every region that contains Python code (`bot/`, `pipeline/`, tests, notebooks, etc.).

For case-directory submit structure see `.claude/rules/bot/pipeline.md`. For pytest conventions see `.claude/rules/bot/tests.md`.

`pyproject.toml` / `uv.lock` / `.python-version` sit at `bot/` root, and `uv run ...` commands are expected to execute from `bot/`.

## Bot Module Architecture (`bot/src/**`)

`bot/src/` holds the shared **development** libraries (not submitted to Kaggle).
Each subdirectory is exposed as a top-level package via `[tool.hatch.build.targets.wheel] packages`, so imports are bare (`from utils.repo_root import ...`, `from dataset...`):

```
bot/src/
  submit/          Kaggle submission packaging / validation / quota (python -m submit)
  dataset/         Selfplay & Kaggle-episode data pipeline (python -m dataset)
  evaluate/        Cross-case eval (metrics, vs_baseline, snapshot_update)
  utils/           Shared utilities (repo_root, fleet kinematics, trajectory safety, gpu_bench)
  gpu/             GPU provider CLIs (one subpackage per provider):
    vast/          Vast.ai GPU pod control CLI (python -m gpu.vast)
    runpod/        RunPod GPU pod control CLI (python -m gpu.runpod)
    kaggle/        Kaggle Notebook GPU training CLI (python -m gpu.kaggle)
```

Agent implementations live under `bot/pipeline/<family>/case<N>/` (see
`.claude/rules/bot/pipeline.md`). The simulator backends and env adapter live
**outside `bot/`** under `simulator/` (`orbit_wars_vendor`, `orbit_wars_rust`,
`orbit_wars_jax`, `orbit_wars_sim`) and are consumed as editable packages.

### Module Design Principles

- Each module owns a single responsibility
- Express inter-module dependencies via explicit imports
- Keep feature extraction and policies loosely coupled so they can be swapped
- Minimize dependencies in the submission entrypoint (`pipeline/<family>/case<N>/main.py`); avoid heavy imports unavailable in the Kaggle runtime

## General Principles

- Comply with PEP 8 and write Pythonic code
- Methods should have referential transparency and idempotency
- Return early and keep nesting shallow
- Follow the Single Responsibility Principle
- Keep third-party libraries to a minimum
- Always import at the top of the file
- No backward compatibility concerns — remove unnecessary code
- Minimize lines of code
- Avoid excessive commenting and logging
- Don't implement temporary measures — make fundamental changes
- 200-400 lines per file typical, 800 max
- NEVER mutate objects — always create new instances

## Type Hints & Naming

- Use Python 3.13 standard types (`list[str]`, `str | None` instead of `List`, `Optional`)
- Avoid `Any` type, `cast`, and `type: ignore` comments
- Type hints for all function arguments and return values
- `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)

```python
# GOOD
def select_action(obs: Observation) -> list[Action]:
    ...

# BAD
def select_action(obs) -> Any:
    ...
```

## Numerics & Performance Conventions

- Vectorize board computations (compute relative positions of fleets/planets in bulk via NumPy)
- The per-turn budget is 1 second (`actTimeout=1`); avoid Python loops and dynamic allocation on hot paths
- Cache future positions of orbital planets and comets at initialization time
- Declare magic numbers (`boardSize=100.0`, `sunRadius=10.0`, etc.) as constants
- Use `pathlib.Path` for file paths

```python
# GOOD: vectorized
import numpy as np

def distances(fleets_xy: np.ndarray, planet_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(fleets_xy - planet_xy, axis=1)
```

## Async & Error Handling

- I/O bound operations (replay persistence, API access): use `async`/`await`
- Parallel self-play: `asyncio.gather` or `multiprocessing.Pool`
- Define appropriate exception classes
- Output structured logs
- Use exception chaining (`raise ... from e`)

```python
class ObservationParseError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to parse observation: {reason}")
```

## Logging

- Use structured logging with JSON format
- Exclude sensitive information (API tokens)
- Use `logging.getLogger(__name__)`
- NEVER use `print()` for logging (do not pollute stdout, even from the submission entrypoint)

## Lint/Formatting

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy .
```

## Notebooks (`.ipynb`)

- Notebooks follow the same rules above. Even for EDA / experimentation, prefer `logging` over `print()`, or use the cell's last-expression evaluation for inline display
- Extract finished logic into `.py` modules. Do not let business logic linger long in notebooks
- Clear cell outputs before committing; do not push large execution results into git

## Code Quality Checklist

- [ ] Code is readable and well-named
- [ ] Functions are small (<50 lines), files are focused (<800 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Proper error handling with exception chaining
- [ ] No `print()` statements — use structured logging
- [ ] No hardcoded values
- [ ] No mutation (immutable patterns used)
- [ ] Type hints for all functions (no `Any`)
- [ ] `ruff format`, `ruff check`, `mypy` pass
