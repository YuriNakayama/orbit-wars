---
name: Testing
description: TDD workflow, unit/integration/E2E testing patterns, eval-driven development, and coverage verification
activation: Writing tests, implementing features with TDD, running E2E tests, or evaluating AI-driven development
---

# Testing

TDD ワークフロー、テストパターン、E2E テスト、eval 駆動開発。

## Commands

- `/tdd` - TDD ワークフロー（RED→GREEN→REFACTOR）
- `/test-coverage` - カバレッジ分析と不足テスト生成
- `/playwright` - Playwright E2E テストの生成・実行
- `/eval-driven` - eval 駆動開発（定義・チェック・レポート）

## TDD Workflow

### Core Principles

1. **Tests BEFORE Code** - 必ずテストを先に書く
2. **Coverage 80%+** - unit + integration + E2E
3. **Test Behavior, Not Implementation** - ユーザー視点

### Cycle

1. Write failing test (RED)
2. Write minimal code to pass (GREEN)
3. Refactor while tests stay green (REFACTOR)
4. Verify coverage

## Unit Tests (Jest/Vitest)

```typescript
import { render, screen, fireEvent } from '@testing-library/react'

describe('Button', () => {
  it('calls onClick when clicked', () => {
    const handleClick = jest.fn()
    render(<Button onClick={handleClick}>Click</Button>)
    fireEvent.click(screen.getByRole('button'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })
})
```

## Integration Tests (API)

```typescript
describe('GET /api/markets', () => {
  it('returns markets successfully', async () => {
    const request = new NextRequest('http://localhost/api/markets')
    const response = await GET(request)
    const data = await response.json()
    expect(response.status).toBe(200)
    expect(data.success).toBe(true)
  })

  it('validates query parameters', async () => {
    const request = new NextRequest('http://localhost/api/markets?limit=invalid')
    const response = await GET(request)
    expect(response.status).toBe(400)
  })
})
```

## E2E Tests (Playwright)

```typescript
import { test, expect } from '@playwright/test'

test('user can search markets', async ({ page }) => {
  await page.goto('/')
  await page.fill('input[placeholder="Search"]', 'election')
  await page.waitForTimeout(600)
  const results = page.locator('[data-testid="market-card"]')
  await expect(results.first()).toContainText('election', { ignoreCase: true })
})
```

### E2E Capabilities

- クロスブラウザテスト実行
- スクリーンショット・動画・トレースキャプチャ
- フレーキーテストの検出と隔離
- CI/CD パイプラインとの統合

## Mocking

```typescript
// Supabase
jest.mock('@/lib/supabase', () => ({
  supabase: {
    from: jest.fn(() => ({
      select: jest.fn(() => ({
        eq: jest.fn(() => Promise.resolve({ data: [{ id: 1 }], error: null }))
      }))
    }))
  }
}))

// External API
jest.mock('@/lib/openai', () => ({
  generateEmbedding: jest.fn(() => Promise.resolve(new Array(1536).fill(0.1)))
}))
```

## Coverage

```bash
npm run test:coverage
# or
pytest --cov=src --cov-fail-under=80
```

| Code Type | Target |
|-----------|--------|
| Critical business logic | 100% |
| Public APIs | 90%+ |
| General code | 80%+ |
| Generated code | Exclude |

## Test Organization

```
src/
  components/Button/
    Button.tsx
    Button.test.tsx
  app/api/markets/
    route.ts
    route.test.ts
e2e/
  markets.spec.ts
  auth.spec.ts
```

## Eval-Driven Development (EDD)

Evals = AI 開発のためのユニットテスト。

### Eval Types

- **Capability** - 新機能が正しく動作するか
- **Regression** - 既存機能が壊れていないか

### Grader Types

- **Code-based** - 自動判定（assert, schema validation）
- **Model-based** - LLM による品質評価
- **Human** - 人間による評価

### Metrics

- `pass@k` - k回中1回以上成功する確率
- `pass^k` - k回全て成功する確率（信頼性指標）

## Common Mistakes

- Testing implementation details → test user-visible behavior
- Brittle selectors (`.css-xyz`) → semantic selectors (`[data-testid]`)
- Test interdependency → each test sets up own data
- `assert True` → meaningful assertions
- `sleep()` in tests → use mocks/waitFor

## Best Practices

1. Write tests first (TDD)
2. One assert per test
3. Descriptive test names: `test_<function>_<scenario>_<expected>`
4. AAA pattern (Arrange-Act-Assert)
5. Mock external dependencies
6. Test edge cases (null, empty, boundary)
7. Test error paths
8. Keep tests fast (< 50ms each)
9. No shared mutable state
10. Review coverage reports
