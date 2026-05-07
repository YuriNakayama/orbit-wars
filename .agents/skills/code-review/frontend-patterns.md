---
name: Frontend
description: TypeScript/React/Next.js coding standards, component patterns, state management, performance optimization, and accessibility
activation: Writing, reviewing, or refactoring TypeScript/React/Next.js code
---

# Frontend

TypeScript/React/Next.js のコーディング規約、パターン、パフォーマンス最適化。

## Commands

- `/code-review` - コード品質・セキュリティレビュー

## Code Quality Principles

- **Readability First** - 自己文書化コード優先
- **KISS** - 最もシンプルな解法
- **DRY** - 共通ロジックの抽出
- **YAGNI** - 不要な機能を作らない

## TypeScript Standards

```typescript
// Descriptive naming (verb-noun)
async function fetchMarketData(marketId: string): Promise<Market> { }
function isValidEmail(email: string): boolean { }

// Proper types (never use 'any')
type Market = {
  id: string
  name: string
  status: 'active' | 'resolved' | 'closed'
}

// Immutability (CRITICAL)
const updatedUser = { ...user, name: 'New Name' }
const updatedArray = [...items, newItem]

// Parallel execution
const [users, markets] = await Promise.all([fetchUsers(), fetchMarkets()])
```

## React Component Patterns

### Composition

```typescript
type ButtonProps = {
  children: React.ReactNode
  onClick: () => void
  variant?: 'primary' | 'secondary'
}

export function Button({ children, onClick, variant = 'primary' }: ButtonProps) {
  return (
    <button onClick={onClick} className={`btn btn-${variant}`}>
      {children}
    </button>
  )
}
```

### Compound Components

```typescript
const TabsContext = createContext<TabsContextValue | undefined>(undefined)

export function Tabs({ children, defaultTab }: { children: React.ReactNode; defaultTab: string }) {
  const [activeTab, setActiveTab] = useState(defaultTab)
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      {children}
    </TabsContext.Provider>
  )
}
```

### Custom Hooks

```typescript
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(handler)
  }, [value, delay])
  return debouncedValue
}
```

## State Management

```typescript
// Context + Reducer for complex state
type Action =
  | { type: 'SET_ITEMS'; payload: Item[] }
  | { type: 'SET_LOADING'; payload: boolean }

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'SET_ITEMS':
      return { ...state, items: action.payload }
    case 'SET_LOADING':
      return { ...state, loading: action.payload }
    default:
      return state
  }
}

// Functional updates for prev-state dependency
setCount(prev => prev + 1)
```

## Performance Optimization

```typescript
// Memoize expensive computations
const sorted = useMemo(() => items.sort((a, b) => b.score - a.score), [items])

// Memoize callbacks
const handleSearch = useCallback((q: string) => setQuery(q), [])

// React.memo for pure components
export const Card = React.memo<CardProps>(({ item }) => <div>{item.name}</div>)

// Lazy loading
const HeavyChart = lazy(() => import('./HeavyChart'))
```

### Virtualization

```typescript
import { useVirtualizer } from '@tanstack/react-virtual'

export function VirtualList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
    overscan: 5,
  })
  // render virtual items...
}
```

## Form Handling

```typescript
import { z } from 'zod'

const Schema = z.object({
  name: z.string().min(1).max(200),
  email: z.string().email(),
})

// Controlled form with validation
const [formData, setFormData] = useState<FormData>(initial)
const [errors, setErrors] = useState<FormErrors>({})
```

## Error Boundary

```typescript
export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }
  render() {
    if (this.state.hasError) return <ErrorFallback error={this.state.error} />
    return this.props.children
  }
}
```

## Accessibility

- Keyboard navigation: `onKeyDown` with ArrowUp/Down/Enter/Escape
- Focus management: save/restore focus on modal open/close
- ARIA attributes: `role`, `aria-expanded`, `aria-modal`

## API Design (Next.js Routes)

```typescript
// Consistent response format
type ApiResponse<T> = {
  success: boolean
  data?: T
  error?: string
  meta?: { total: number; page: number; limit: number }
}

// Input validation with Zod
const validated = CreateSchema.parse(body)
```

## Conditional Rendering

```typescript
// Clear conditionals (not ternary chains)
{isLoading && <Spinner />}
{error && <ErrorMessage error={error} />}
{data && <DataDisplay data={data} />}
```

## File Organization

```
src/
  app/           Next.js App Router
  components/    React components (PascalCase)
  hooks/         Custom hooks (useXxx.ts)
  lib/           Utilities (camelCase)
  types/         TypeScript types
```

## Code Smells to Avoid

- Functions > 50 lines → split
- Nesting > 4 levels → early returns
- Magic numbers → named constants
- `any` type → proper types
- Mutation → spread operator
