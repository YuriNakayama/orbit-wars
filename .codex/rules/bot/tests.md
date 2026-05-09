---
paths:
  - "bot/tests/**"
---

# Bot Test Rules (`bot/tests/**`)

Pytest conventions for the bot test suite. General Python rules (type hints, logging, naming) follow the repository Python rules.

## Frameworks

- **Unit/Integration**: Pytest + pytest-asyncio
- Tests mirror `src/` structure in `tests/`

## Test Guidelines

- Write in AAA pattern (Arrange, Act, Assert)
- Use Fixtures for common setup
- Minimize use of mock and patch — keep close to actual behavior
- Each test should be executable independently
- Tests live under `bot/tests/`, mirroring the `bot/src/` layout
- For agent tests, build scenarios through `env.orbit_wars.make_orbit_wars_env()` so local runs use the in-repo simulator backend.

```python
import pytest
from env.orbit_wars import make_orbit_wars_env, run_orbit_wars_episode

@pytest.fixture
def env():
    return make_orbit_wars_env(debug=True)

def test_agent_does_nothing_is_legal(env):
    def noop(obs):
        return []

    run_orbit_wars_episode(env, [noop, noop])
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

## Test Classification Layout

When reorganizing `bot/tests`, classify tests by execution scope first, then keep only `src` or `pipeline` directly under each class directory:

```text
bot/tests/
  unit/
    src/
    pipeline/
  integration/
    src/
    pipeline/
  e2e/
    pipeline/
```

Prefer placing as many tests as practical under `unit`.

- **unit**: ordinary unit tests. Use this for pure logic and small APIs, including geometry, physics, decoder, featurizer, model forward, metrics, schema/dataclass conversion, parser/validation, small `tmp_path` IO checks, small mocks, dataset `__len__`/`__getitem__`, and DataLoader shape checks.
- **integration**: smoke tests that treat internal logic as a black box. Use this for training smoke tests, CLI happy paths, packaging/archive smoke tests, external-service mock flows, launch/watch flows, and other "does this workflow minimally run" checks.
- **e2e**: self-play and real Orbit Wars episode execution. Any test that creates/runs the Orbit Wars environment (`make_orbit_wars_env()`, `run_orbit_wars_episode(...)`, `env.step(...)`), runs agents through an episode, performs self-play, or evaluates win/loss against a baseline belongs under `e2e/pipeline`.

Additional placement rules:

- `src/evaluation`'s own logic tests belong under `unit/src/evaluation`.
- Tests that use `src/evaluation` to evaluate a pipeline case belong under `e2e/pipeline/...` when they execute self-play or Orbit Wars episodes.
- Prefer `conftest.py` for shared fixtures and pytest-only setup. Use a `utils` package only for tiny importable helpers that cannot be represented as fixtures.
- `utils` packages must not contain `test_*.py` files or `test_*` test functions. Test bodies must live outside `utils`.
- Tests may depend on `tests/pipeline/utils`-style helpers only when the helper is tiny (for example short assertion helpers). If helper logic becomes worth testing, move it to `bot/src/evaluation` or another `bot/src` module.
- Avoid importing test implementation from another test module. Shared production-like evaluation, aggregation, snapshot, or comparison logic belongs in `bot/src/evaluation`.
- Pytest scope markers are assigned from the top-level directory by `bot/tests/conftest.py` only for `integration` and `e2e`; `unit` tests intentionally have no scope marker. Keep the directory classification as the source of truth and reserve per-test markers for orthogonal concerns such as `slow` or `timeout`.

