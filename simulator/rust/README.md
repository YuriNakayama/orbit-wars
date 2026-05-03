# `simulator/rust/` — Rust port of Orbit Wars

Drop-in replacement for the `interpreter` function from
`kaggle_environments.envs.orbit_wars`, exposed to Python via
[PyO3](https://pyo3.rs/) + [maturin](https://www.maturin.rs/), plus a
unified `run(...)` helper that covers single-match, in-process serial,
and multi-process parallel self-play with one function call.

## Layout

```
simulator/rust/
├── Cargo.toml                    # crate config (pyo3, rand_chacha, criterion)
├── pyproject.toml                # maturin build config (module = orbit_wars_rust._lib)
├── src/
│   ├── lib.rs                    # #[pymodule] entry, GIL release wrapper
│   ├── state.rs                  # OrbitWarsState / Planet / Fleet / CometGroup
│   ├── rng.rs                    # ChaCha12 wrapper helpers
│   ├── physics.rs                # fleet motion, planet rotation, sweep collision
│   ├── combat.rs                 # combat resolution
│   ├── generation.rs             # generate_planets / generate_comet_paths
│   ├── interpreter.rs            # step() — orchestrates phases
│   ├── fast_helpers.rs           # PyO3 helpers (fast_filter_expired, etc.)
│   └── pybind.rs                 # PyDict <-> OrbitWarsState conversions
├── python/
│   └── orbit_wars_rust/          # Python facade (kaggle_environments.register hook)
│       ├── __init__.py
│       └── _facade.py
├── tests/                        # cargo test (Rust unit + integration)
└── benches/
    └── step.rs                   # criterion benchmark
```

## Build & install

From the repo root:

```bash
(cd simulator/rust && uv run maturin develop --release)
```

This builds the Rust extension and installs it into the **current venv** as
`orbit_wars_rust._lib`. Importing `orbit_wars_rust` from any Python code in
the same venv automatically triggers
`kaggle_environments.register("orbit_wars", ...)`.

## Running tests

```bash
# Rust-side
(cd simulator/rust && cargo test)
(cd simulator/rust && cargo fmt --check && cargo clippy -- -D warnings)

# Python-side (via bot venv)
(cd bot && uv run pytest ../simulator/rust/python/tests)

# Benchmark suite (slow, prints speedup ratios)
(cd bot && uv run pytest ../simulator/rust/python/tests/test_benchmark.py -s -m slow)
```

## Recommended API: `orbit_wars_rust.run(...)`

A single function covers every speed tier — pick by argument. The default
backend is `"rust"` (the Python `env.run` path is preserved unchanged for
callers that import `kaggle_environments.make("orbit_wars", ...)` directly).

```python
import orbit_wars_rust

# 1 match (returns one summary dict)
result = orbit_wars_rust.run(["random", "random"], seed=0)
# → {"seed": 0, "turns": 500, "rewards": [-1, 1], "statuses": ["DONE", "DONE"]}

# N matches, in-process serial
results = orbit_wars_rust.run(["random", "random"], seeds=range(30))

# N matches across worker processes (Pool helper is built in)
results = orbit_wars_rust.run(
    ["random", "random"],
    seeds=range(30),
    parallel=8,
    mp_context="fork",   # default "spawn" (PyTorch/CUDA safe)
)

# Force the upstream Python interpreter (parity / debugging)
result = orbit_wars_rust.run(["random", "random"], seed=0, backend="python")
```

### Argument cheatsheet

| Argument | Default | Effect |
|---|---|---|
| `agents`              | required | List of registered agent names (e.g. `"random"`, `"starter"`). |
| `seed=N`              | —        | Run a single match. Mutually exclusive with `seeds=`. |
| `seeds=range(N)`      | —        | Run N matches; one summary dict per seed. |
| `parallel=K`          | `1`      | Run across K worker processes (`1` = in-process serial). |
| `mp_context="spawn"`  | `"spawn"` | `"spawn"` is safe with PyTorch/CUDA tensors held by the parent; `"fork"` skips the per-worker import tax (~0.5–1 s). |
| `backend="rust"`      | `"rust"` | Rust interpreter (default) or `"python"` (parity / debugging). |

### Speed tiers (30 matches × 2 random agents, M-series Mac, 12 cores)

| Call | wall-clock | speedup vs Python |
|---|---:|---:|
| `run(..., backend="python")` | 51 s | 1.0× (baseline) |
| `run(..., seeds=range(30))` (default backend, serial) | 1.8–1.9 s | **~27×** |
| `run(..., seeds=range(30), parallel=4, mp_context="fork")` | 0.47 s | **~108×** |
| `run(..., seeds=range(30), parallel=8, mp_context="fork")` | 0.36–0.43 s | **~120–141×** |
| `run(..., seeds=range(30), parallel=12, mp_context="fork")` | 0.28 s | **~180×** |

Throughput stays roughly linear with `parallel` until startup cost is
amortized. For larger N the scaling tightens — N=120 / parallel=12 / fork
hits ~92 matches/sec ≈ 154× over the Python baseline.

## Lower-level alternatives

When the caller needs the full `env.steps` history (replay JSON, per-frame
inspection) or wants to reuse a `make()`-built env directly, drop down
to one of:

```python
from kaggle_environments import make
import orbit_wars_rust

orbit_wars_rust.use_rust()                          # backend switch (1×–2× transparent)
env = make("orbit_wars", configuration={"agents": 2})
env.run(["random", "random"])                        # transparent path

orbit_wars_rust.run_episode(env, ["random", "random"])             # 1 match, full env.steps
orbit_wars_rust.run_episodes(env, ["random", "random"], range(30)) # reuse env across seeds
orbit_wars_rust.run_episodes_parallel(                              # raw Pool helper
    agents=["random", "random"], seeds=range(30),
    parallel=8, mp_context="fork",
)
```

These predate the unified `run()` and remain available for backward
compatibility. New code should prefer `run()`.

## Backend selection (rarely needed when using `run`)

```python
import orbit_wars_rust

orbit_wars_rust.use_rust()                  # opt into Rust
orbit_wars_rust.use_python()                # back to upstream Python
orbit_wars_rust.set_backend("rust")         # explicit setter
orbit_wars_rust.get_backend()               # → "python" or "rust"

with orbit_wars_rust.backend("rust"):
    env.run([...])                          # scoped switch with auto-restore
```

The active backend is read on every `env.step`, so toggling it takes
effect immediately without re-importing. Unknown backend names raise
`ValueError`. `run(..., backend=...)` sets this internally.

## Constraints worth knowing

- **`run()` agents must be registered names (str).** Custom callables
  cannot cross process boundaries via pickle. For in-process callables,
  build the env yourself and call `run_episode(env, callable_agents)`.
- **`mp_context="fork"`** is unsafe when the parent holds CUDA tensors,
  PyTorch DataLoader workers, or other thread-bound state. Use
  `"spawn"` (default) in those cases — `parallel >= 4` only pays off
  when N ≳ 100 because spawn imports `kaggle_environments` per worker
  (~1 s startup cost).
- **Parity is bit-exact** vs the upstream Python interpreter
  (`rel_tol=1e-9`). See `python/tests/test_parity.py`.
- **Kaggle submission path is unchanged** — submitted bots use
  `kaggle_environments.make(...).run(...)` directly and bypass the
  `run()` helper entirely.
