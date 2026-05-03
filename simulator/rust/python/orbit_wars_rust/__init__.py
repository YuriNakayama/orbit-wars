"""Rust-backed Orbit Wars simulator.

Importing this package registers `orbit_wars` with `kaggle_environments` so
that `kaggle_environments.make("orbit_wars", ...)` dispatches through the
backend-selection facade.

## Default behavior: Python

Without configuring anything, the facade dispatches to the **upstream Python
interpreter** (the Apache-2.0 vendored copy in `simulator/python/`).
Importing this package therefore preserves the exact behavior every existing
call site already sees.

## Switching backends from Python code

Re-exported helpers from `_facade`:

    import orbit_wars_rust
    orbit_wars_rust.use_rust()                  # opt into Rust
    orbit_wars_rust.use_python()                # back to upstream Python
    orbit_wars_rust.set_backend("rust")         # explicit setter
    orbit_wars_rust.get_backend()               # → "python" or "rust"

    with orbit_wars_rust.backend("rust"):
        env.run([...])                          # scoped switch with restore

    # Optional: disable the Rust-side comet path generator (only relevant
    # while backend == "rust"). Falls back to upstream Python comet spawn.
    orbit_wars_rust.set_rust_comet(False)

The active backend is read on every `env.step`, so toggling it takes effect
immediately without re-importing.
"""

from __future__ import annotations

from kaggle_environments import register

from . import _facade, _patches
from ._facade import (
    Backend,
    backend,
    environment_dict,
    get_backend,
    get_rust_comet,
    set_backend,
    set_rust_comet,
    use_python,
    use_rust,
)

_patches.apply()
register("orbit_wars", environment_dict())

__all__ = [
    "Backend",
    "_facade",
    "_patches",
    "backend",
    "environment_dict",
    "get_backend",
    "get_rust_comet",
    "set_backend",
    "set_rust_comet",
    "use_python",
    "use_rust",
]
