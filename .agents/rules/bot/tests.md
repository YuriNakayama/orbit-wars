---
paths:
  - "bot/tests/**"
---

# Bot Test Rules (`bot/tests/**`)

Pytest conventions for the bot test suite. General Python rules (type hints, logging, naming) live in `.agents/rules/python.md`.

## Frameworks

- **Unit/Integration**: Pytest + pytest-asyncio
- Tests mirror `src/` structure in `tests/`

## Test Guidelines

- Write in AAA pattern (Arrange, Act, Assert)
- Use Fixtures for common setup
- Minimize use of mock and patch — keep close to actual behavior
- Each test should be executable independently
- Tests live under `bot/tests/`, mirroring the `bot/src/` layout
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

## Test-Driven Development

1. Write test first (RED) — test should FAIL
2. Write minimal implementation (GREEN) — test should PASS
3. Refactor (IMPROVE)
4. Verify coverage (80%+)

## Running tests

```bash
# All tests with coverage
dev/test-bot

# Specific module
uv run --directory bot pytest tests/<path> -x --no-header -q
```
