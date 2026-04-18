---
paths:
  - "backend/**"
---

# Backend Rules

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

データエージェントの共通ライブラリとして以下のモジュール構成を想定:

```
backend/src/
  planner/         質問分解・実行計画生成
  executor/        ツール選択・実行（Python, SQL, API）
  reasoner/        中間結果の推論・統合
  integrator/      最終回答の合成
  parsers/         データパーサー（CSV, JSON, PDF, DOCX, PNG）
  utils/           共通ユーティリティ
```

### モジュール設計原則

- 各モジュールは単一責任を持つ
- モジュール間の依存は明示的にインポートで表現
- パーサーはデータ形式ごとに独立して実装
- 共通インターフェースを定義し、拡張可能に設計

## Type Hints & Naming

- Use Python 3.12 standard types (`list[str]`, `str | None` instead of `List`, `Optional`)
- Avoid `Any` type, `cast`, and `type: ignore` comments
- Type hints for all function arguments and return values
- `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)

```python
# GOOD
def get_user(user_id: str) -> User | None:
    ...

# BAD
def get_user(user_id) -> Any:
    ...
```

## Data Processing Conventions

- DataFrameの操作はメソッドチェーンを活用し、中間変数を減らす
- SQLクエリはパラメータ化して実行（SQLインジェクション防止）
- 大規模データはチャンク処理でメモリ効率を確保
- ファイルパスは `pathlib.Path` を使用

```python
# GOOD: パラメータ化クエリ
import duckdb

conn = duckdb.connect(db_path)
result = conn.execute(
    "SELECT * FROM sales WHERE region = ?", [region]
).fetchdf()

# BAD: 文字列結合
result = conn.execute(f"SELECT * FROM sales WHERE region = '{region}'").fetchdf()
```

## Async & Error Handling

- I/O bound operations: use `async`/`await`
- Parallel processing: use `asyncio.gather`
- Define appropriate exception classes
- Output structured logs
- Use exception chaining (`raise ... from e`)

```python
class DataParseError(Exception):
    def __init__(self, file_path: str, reason: str) -> None:
        super().__init__(f"Failed to parse {file_path}: {reason}")
        self.file_path = file_path

async def parse_data(file_path: str) -> DataFrame:
    try:
        return await _read_file(file_path)
    except FileNotFoundError as e:
        raise DataParseError(file_path, "file not found") from e
```

## Logging

- Use structured logging with JSON format
- Exclude sensitive information (tokens, passwords, PII)
- Use `logging.getLogger(__name__)` or project logger
- NEVER use `print()` for logging

## Lint/Formatting

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy .
```

## Testing

### Frameworks

- **Unit/Integration**: Pytest + pytest-asyncio
- Tests mirror `src/` structure in `backend/tests/`

### Test Guidelines

- Write in AAA pattern (Arrange, Act, Assert)
- Use Fixtures for common setup
- Minimize use of mock and patch — keep close to actual behavior
- Each test should be executable independently
- データパーサーのテストにはサンプルデータファイルを `tests/fixtures/` に配置

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("region,amount\neast,100\nwest,200\n")
    return csv_file

def test_parse_csv(sample_csv: Path):
    # Arrange
    parser = CsvParser()

    # Act
    result = parser.parse(sample_csv)

    # Assert
    assert len(result) == 2
    assert result.columns.tolist() == ["region", "amount"]
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
- [ ] SQL queries are parameterized
- [ ] `ruff format`, `ruff check`, `mypy` pass
