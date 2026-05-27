# orbit_wars_sim

Backend-agnostic adapter for running the Orbit Wars environment. It picks a
simulator backend at runtime and registers it with `kaggle_environments`:

| `ORBIT_WARS_BACKEND` | Backend | Source |
|----------------------|---------|--------|
| (unset, default)     | Rust    | `simulator/rust` → `orbit_wars_rust` (PyO3 native ext) |
| `python`             | Python  | `simulator/python` → `orbit_wars_vendor` (vendored upstream sim) |

The Rust backend is the fast default; the Python backend is a fallback for
hosts without a pre-built native wheel (e.g. Kaggle Kernel).

## Design

This package is **independent of `bot/`**. It locates the sibling backend dirs
(`rust/`, `python/`) relative to its own `__file__` (three parents up to
`simulator/`), so it works from any worktree without a repo-root walk.

## Import

```python
from orbit_wars_sim import make_orbit_wars_env, run_orbit_wars_episode

env = make_orbit_wars_env(seed=42, agents=4)
steps = run_orbit_wars_episode(env, [agent_a, agent_b, agent_c, agent_d])
```

`orbit_wars_rust` (default) and `orbit_wars_vendor` (`python` backend) are
loaded lazily via `sys.path`, so the native Rust wheel is an optional runtime
dependency, not a build/install dependency.
