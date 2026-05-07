---
name: Backend
description: Python/FastAPI backend patterns, architecture, testing, build fixes, and code review. Covers API design, async, DI, caching, error handling, and auth.
activation: Writing, reviewing, testing, or fixing Python/FastAPI backend code
---

# Backend

Python/FastAPI バックエンド開発の包括的スキル。パターン、アーキテクチャ、テスト、ビルド修正、コードレビュー。

## Commands

- `/python-tdd` - TDD ワークフロー（RED→GREEN→REFACTOR）
- `/python-lint` - ビルド・型・lint エラーの段階的修正
- `/python-review` - コードレビュー（パターン、async安全性、セキュリティ）
- `/typescript-lint` - TypeScript/ビルド・lintエラーの段階的修正

## Core Principles

1. **Readability Counts** - PEP 8 準拠
2. **Explicit is Better than Implicit** - 明確なコード
3. **Composition over Inheritance** - Protocol パターン優先
4. **Type Safety** - 全関数に型ヒント

## Type Hints (Python 3.10+)

```python
def process(data: list[dict[str, int]], flag: bool = False) -> str | None:
    ...

type UserId = int
type Callback = Callable[[str, int], bool]
```

## FastAPI API Design

```python
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

@router.get("/", response_model=list[SessionResponse])
async def list_sessions(
    service: SessionService = Depends(get_session_service),
) -> list[SessionResponse]:
    return await service.list_all()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    return await service.create(request)
```

## Pydantic Models

```python
from pydantic import BaseModel, EmailStr, Field

class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(ge=0, le=150)

@dataclass(frozen=True)
class User:
    name: str
    email: str
    age: int = 0
```

## Repository Pattern

```python
@runtime_checkable
class Repository(Protocol):
    async def find_by_id(self, id: str) -> dict | None: ...
    async def save(self, entity: dict) -> dict: ...
    async def delete(self, id: str) -> None: ...
```

## Service Layer & Dependency Injection

```python
@dataclass
class UserService:
    repository: Repository
    cache: CacheClient

    async def create_user(self, data: CreateUserRequest) -> User:
        user = User.from_request(data)
        saved = await self.repository.save(user.to_dict())
        await self.cache.set(f"user:{saved['id']}", saved)
        return User.from_dict(saved)

# FastAPI DI
def get_user_service() -> UserService:
    return UserService(repository=PostgresRepository(), cache=RedisCache())
```

## Error Handling

```python
class DomainError(Exception):
    """Base exception for domain errors."""

class UserNotFoundError(DomainError):
    def __init__(self, user_id: str) -> None:
        super().__init__(f"User not found: {user_id}")
        self.user_id = user_id

# Re-raise with context
try:
    result = external_api.call()
except ConnectionError as e:
    raise ServiceUnavailableError("External API unreachable") from e

# FastAPI exception handler
@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})
```

## Async/Await Patterns

```python
# Concurrent execution
async def fetch_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Semaphore for rate limiting
async def rate_limited_fetch(urls: list[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        async def fetch(url: str):
            async with semaphore:
                return await client.get(url)
        return await asyncio.gather(*[fetch(url) for url in urls])
```

## Context Managers

```python
@contextmanager
def database_transaction(session):
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

## Caching

```python
class CachedRepository:
    def __init__(self, base_repo: Repository, redis: Redis) -> None:
        self.base_repo = base_repo
        self.redis = redis

    async def find_by_id(self, id: str) -> dict | None:
        cached = await self.redis.get(f"item:{id}")
        if cached:
            return json.loads(cached)
        item = await self.base_repo.find_by_id(id)
        if item:
            await self.redis.setex(f"item:{id}", 300, json.dumps(item))
        return item
```

## Retry with Backoff

```python
async def fetch_with_retry(fn, max_retries: int = 3):
    for i in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(2 ** i)
```

## Middleware (FastAPI)

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    token = request.headers.get("authorization", "").replace("Bearer ", "")
    if not token:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    request.state.user = verify_token(token)
    return await call_next(request)
```

## Testing (pytest)

```python
# TDD: RED → GREEN → REFACTOR
@pytest.mark.parametrize("email,expected", [
    ("user@example.com", True),
    ("invalid", False),
    ("", False),
])
def test_validate_email(email: str, expected: bool) -> None:
    assert validate_email(email) == expected

# Fixtures
@pytest.fixture
def db_session():
    session = create_test_session()
    yield session
    session.rollback()
    session.close()

# Async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

### Coverage

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

## Immutability

```python
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    debug: bool = False

new_config = replace(config, port=9090)  # New object
```

## Logging & Configuration

```python
# Structured logging
log = structlog.get_logger()
log.info("processing_user", user_id=user_id, action="create")

# Configuration
class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    debug: bool = False
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

## Tooling

```bash
ruff format .                         # Formatter
ruff check . --fix                    # Linter
mypy . --strict                       # Type checker
bandit -r src/                        # Security
pytest --cov=src --cov-fail-under=80  # Tests + coverage
```

## Python Idioms

| Instead of | Use |
|---|---|
| `if len(x) == 0` | `if not x` |
| `for i in range(len(lst))` | `for i, v in enumerate(lst)` |
| `str + str` in loop | `"".join(parts)` |
| `os.path.join()` | `pathlib.Path()` |
| `type(x) == int` | `isinstance(x, int)` |
| Manual dict merge | `d1 \| d2` |
| Class with only `__init__` | `@dataclass` |

## Anti-Patterns

- Mutable default arguments (`def f(x=[])`)
- Bare `except:` clauses
- Global mutable state
- Star imports (`from x import *`)
- `print()` instead of logging
- `eval()` or `exec()`
