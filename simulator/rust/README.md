# `simulator/rust/` — Rust port of Orbit Wars

Drop-in replacement for the `interpreter` function from
`kaggle_environments.envs.orbit_wars`, exposed to Python via
[PyO3](https://pyo3.rs/) + [maturin](https://www.maturin.rs/).

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
the same venv will then trigger `kaggle_environments.register("orbit_wars",
...)` automatically (see Step 11 of the implementation plan).

## Running tests

```bash
# Rust-side
(cd simulator/rust && cargo test)
(cd simulator/rust && cargo fmt --check && cargo clippy -- -D warnings)

# Python-side (via bot venv)
(cd bot && uv run pytest ../simulator/rust/python/tests)
```

## Backend selection at runtime

The default backend is the upstream Apache-2.0 vendored Python interpreter,
so existing call sites preserve their pre-existing behavior. Opt into Rust
explicitly via the public Python API:

```python
import orbit_wars_rust

orbit_wars_rust.use_rust()                  # opt into Rust
orbit_wars_rust.use_python()                # back to upstream Python
orbit_wars_rust.set_backend("rust")         # explicit setter
orbit_wars_rust.get_backend()               # → "python" or "rust"

with orbit_wars_rust.backend("rust"):
    env.run([...])                          # scoped switch with auto-restore
```

The active backend is read on every `env.step`, so toggling it takes effect
immediately without re-importing. Unknown backend names raise `ValueError`.
