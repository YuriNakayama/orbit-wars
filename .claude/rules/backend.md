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

Orbit Wars エージェントの共通ライブラリとして以下のモジュール構成を想定:

```
backend/src/
  agents/          提出用エージェント（Kaggle Submission entrypoint）
  env/             kaggle-environments ラッパー、自己対戦ユーティリティ
  features/        観測→特徴量、軌道予測、脅威評価
  policies/        ルールベース / 学習済みポリシー
  utils/           共通ユーティリティ（数学・可視化・ロギング）
```

### モジュール設計原則

- 各モジュールは単一責任を持つ
- モジュール間の依存は明示的にインポートで表現
- 特徴量抽出とポリシーは疎結合に保ち、差し替え可能に設計
- Submission (`src/agents/main.py`) は依存を最小化（Kaggle環境にない重い依存は避ける）

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

- 盤面計算はベクトル化する（NumPyで艦・惑星の相対位置を一括計算）
- 1ターン 1秒（`actTimeout=1`）のため、ホットパスでの Python ループ・動的確保を避ける
- 軌道惑星・コメットの未来位置は初期化時にキャッシュ
- マジックナンバー（`boardSize=100.0`, `sunRadius=10.0` 等）は定数として宣言
- ファイルパスは `pathlib.Path` を使用

```python
# GOOD: ベクトル化
import numpy as np

def distances(fleets_xy: np.ndarray, planet_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(fleets_xy - planet_xy, axis=1)
```

## Async & Error Handling

- I/O bound operations (リプレイ保存・APIアクセス): use `async`/`await`
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
- NEVER use `print()` for logging（Submission でも stdout を汚さない）

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
- テストは `backend/tests/` に `backend/src/` の構造をミラーして配置
- エージェントのテストでは `kaggle_environments.make("orbit_wars")` を使ってシナリオを構築

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
