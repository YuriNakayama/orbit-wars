---
paths:
  - "backend/src/**"
  - "backend/tests/**"
---

# Backend Rules

Python の実装は `backend/` 配下に集約されています。`pyproject.toml` / `uv.lock` / `.python-version` は `backend/` 直下にあり、`uv run ...` 系のコマンドは `backend/` で実行することを前提にします。

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

## Module Architecture

The shared library for Orbit Wars agents follows the module layout below:

```
backend/src/
  agents/          Submission agents (Kaggle Submission entrypoint)
  env/             kaggle-environments wrappers and self-play utilities
  features/        Observation -> features, orbit prediction, threat evaluation
  policies/        Rulebase / trained policies
  utils/           Shared utilities (math, visualization, logging)
```

### Module Design Principles

- Each module owns a single responsibility
- Express inter-module dependencies via explicit imports
- Keep feature extraction and policies loosely coupled so they can be swapped
- Minimize dependencies in the submission entrypoint (`src/agents/main.py`); avoid heavy imports unavailable in the Kaggle runtime

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

## Testing

### Frameworks

- **Unit/Integration**: Pytest + pytest-asyncio
- Tests mirror `src/` structure in `tests/`

### Test Guidelines

- Write in AAA pattern (Arrange, Act, Assert)
- Use Fixtures for common setup
- Minimize use of mock and patch — keep close to actual behavior
- Each test should be executable independently
- Tests live under `backend/tests/`, mirroring the `backend/src/` layout
- For agent tests, build scenarios with `kaggle_environments.make("orbit_wars")`

```python
import pytest
from kaggle_environments import make

@pytest.fixture
def env():
    return make("orbit_wars", debug=True)

def test_agent_does_nothing_is_legal(env):
    def noop(obs):
        return []

    env.run([noop, noop])
    assert env.state[0]["status"] in {"DONE", "ACTIVE"}
```

### Test-Driven Development

1. Write test first (RED) — test should FAIL
2. Write minimal implementation (GREEN) — test should PASS
3. Refactor (IMPROVE)
4. Verify coverage (80%+)

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
