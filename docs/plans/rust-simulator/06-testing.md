# Test Strategy

## Testing Approach

3 層 + 1 ストレス層構成で、Rust 単体・Python 互換・e2e parity・パフォーマンスを段階的に保証する:

1. **Rust unit/integration tests** (`simulator/rust/tests/`) — 公式 27 cases を `cargo test` で port。
2. **Python vendored tests** (`simulator/python/orbit_wars_vendor/tests/`) — 公式テストをそのまま回し、参照実装の regression をガード。
3. **Python↔Rust e2e parity** (`simulator/rust/python/tests/test_parity.py`) — 同一 actions・同一初期 state で per-step state hash 比較。
4. **Rust property test (proptest)** — random actions ストレスで edge case を発掘。

すべてのテストは `dev/test-bot` から呼べる。CI には軽量分のみ load し、parity / benchmark は slow marker で nightly に回す。

## Unit Tests

### Rust (`simulator/rust/tests/`, `cargo test`)
公式 `test_orbit_wars.py` の **27 cases を Rust にネイティブ port**。各テストは `tests/test_helpers.rs` で共有された `make_initial_state(planets, fleets)` fixture を使う。

| File | Cases | What it covers |
|------|-------|----------------|
| `test_symmetry.rs` | 4 | symmetry, 4p initialization, diagonal orbiting group, 4p start placement |
| `test_combat.rs` | 9 | simple capture, reinforce, attacker insufficient, two attackers, ties, multi-fleet |
| `test_motion.rs` | 3 | sun collision, leaving board, surviving inside |
| `test_reward.rs` | 7 | max steps, elimination, fleets-only, ties, 4p elimination, fleet ships |
| `test_full_episode.rs` | 1+ | 100 turn full run smoke (final state shape) |
| `test_phase_alignment.rs` | TBD | 各 phase 後の中間 state を Python (vendored) と比較する debug-only tests (`#[cfg(test)]` + Python サブプロセス呼び出し or 事前 dump JSON) |

### Python (`simulator/python/orbit_wars_vendor/tests/`)
公式 `test_orbit_wars.py` を pytest 互換にだけ調整して同梱。

```bash
(cd simulator/python && uv run pytest tests)   # 27 cases green
```

### Python (`simulator/rust/python/tests/`)
- `test_facade.py` — `ORBIT_WARS_BACKEND=python|rust` での `make()` 経路切替が機能していることを確認。
- `test_register.py` — `import orbit_wars_rust` の side-effect で `kaggle_environments.register("orbit_wars", ...)` が走っていることを確認。

## Integration / e2e Tests

### Python↔Rust parity (`simulator/rust/python/tests/test_parity.py`)
- **Setup**: 公式 Python で `env.reset(seed)` を 1 turn 走らせ、`obs.planets / fleets / ...` を取得 → Rust state に注入。
- **Run**: 200 step × 100 episode、各 step で agent action は同一 PRNG seed で生成。
- **Assert**: per-step `observation.{planets,fleets,initial_planets,...}` を JSON 化 → 並べ替え後 deep-compare、float 値は **relative tolerance 1e-9**。
- **Marker**: `@pytest.mark.slow`（CI default では skip、nightly で full run）。

### selfplay smoke (`bot/tests/pipeline/imitation/case1/test_agent_integration.py` 既存)
- 既存テストを Rust backend (default) で再実行し regression なしを確認。
- 旧 Python backend (`ORBIT_WARS_BACKEND=python`) でも同じ green を確認するため、`pytest --backend=python` 等の fixture parameter を追加。

## Stress / Property Tests

### Rust proptest (`simulator/rust/tests/test_property.rs`)
- `proptest = "1"` を dev-deps。
- Strategies:
  - `arb_initial_state()` で planets 5–10 group / fleets 0–50 / step 0–500 を生成。
  - `arb_actions()` で各 player の moves を 0–10 個生成 (from_planet_id, angle, ships)。
- Invariants:
  - `step()` が panic しない。
  - reward が `[-1, 1]` の範囲。
  - `state.planets` の `id` が常にユニーク。
- 1 PR ごとに 1000 cases run（`#[ignore]` で local-only にする選択肢あり）。

### Bench / Benchmark

#### Rust (criterion) — `simulator/rust/benches/parity.rs`
- 200 step × 2 player × 100 反復で per-step latency 測定。
- 期待値: median **50–150µs/step** (M2 Pro)。

#### Python (pytest-benchmark) — `simulator/rust/python/tests/test_benchmark.py`
- `make("orbit_wars", configuration={"agents": 2, "seed": s}); env.run(["random", "random"])` を `ORBIT_WARS_BACKEND=python|rust` 切替で 30 episode 計測。
- `speedup = python_time / rust_time` を計算し `>= 10` を assert（`pytest.warns` で warning に degrade 可能）。

## Test Data

- **公式 27 test fixtures**: vendored copy 内に `simulator/python/orbit_wars_vendor/tests/fixtures/` がある場合はそのまま流用。なければインライン (公式は state を testcase 内で組み立てる方式)。
- **parity baseline state**: ランタイムで Python 公式から生成 (固定 seed)。事前 JSON dump は不要。
- **benchmark seed**: `seed=0..29` を 30 通り回し、平均を取る。

## Coverage Targets

- **Rust** (`cargo tarpaulin --out Stdout` ローカル実行): line coverage **80% 以上**。CI では measure せず警告のみ。
- **Python (`simulator/python/`, `simulator/rust/python/`)**: pytest-cov で **90% 以上** (vendored 内部は元から 100% 寄り、facade は薄い)。
- **`bot/`**: 既存 coverage を維持 (今回変更なし)。

## CI Integration

`.github/workflows/ci-bot.yml` で実行:
```yaml
- run: uv run ruff format --check .
- run: uv run ruff check .
- run: uv run mypy --config-file pyproject.toml .
- run: uv run pytest tests -m "not slow"      # bot
- run: uv run pytest simulator/python/tests -m "not slow"
- run: uv run pytest simulator/rust/python/tests -m "not slow"
- run: cd simulator/rust && cargo fmt --check
- run: cd simulator/rust && cargo clippy -- -D warnings
- run: cd simulator/rust && cargo test
```

Nightly cron (`.github/workflows/nightly-parity.yml` を新設, optional):
```yaml
- run: uv run pytest simulator/rust/python/tests/test_parity.py -m slow
- run: uv run pytest simulator/rust/python/tests/test_benchmark.py -m slow
```

## Manual Verification

PR マージ前の手動検証手順:

```bash
# 1. rename 直後 + simulator/ skeleton
dev/setup
dev/test-bot

# 2. simulator/rust の build と register
(cd simulator/rust && uv run maturin develop --release)
uv run --directory bot python -c "import orbit_wars_rust; from kaggle_environments import make; env = make('orbit_wars', configuration={'agents': 2, 'seed': 42}); env.run(['random', 'random']); print(f'turns={len(env.steps)}, rewards={[s[i].get(\"reward\") for i, s in enumerate(env.steps[-1])]}')"

# 3. Python backend fallback
ORBIT_WARS_BACKEND=python uv run --directory bot python -c "..."   # 同じ結果

# 4. parity (slow)
uv run --directory bot pytest ../simulator/rust/python/tests/test_parity.py -m slow

# 5. self-play 100 マッチ (Rust default)
uv run --directory bot python -m dataset selfplay run --num-matches 100 --no-dvc-add

# 6. benchmark (manual, 10x 検証)
uv run --directory bot pytest ../simulator/rust/python/tests/test_benchmark.py -m slow --benchmark-only
```

期待結果は `docs/plans/rust-simulator/benchmark_results.md` に記録。
