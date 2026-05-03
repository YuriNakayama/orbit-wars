"""Rust-backed Orbit Wars simulator.

Importing this package registers `orbit_wars` with `kaggle_environments` so
that `kaggle_environments.make("orbit_wars", ...)` dispatches through the
backend-selection facade.

## Default behavior: Python

Without setting any environment variable, the facade dispatches to the
**upstream Python interpreter** (the Apache-2.0 vendored copy in
`simulator/python/`). Importing this package therefore preserves the
exact behavior every existing call site already sees.

## Opting into Rust

Set `ORBIT_WARS_BACKEND=rust` before any `env.run` / `env.step` call to
route through the Rust interpreter. See `_facade.py` module docstring
for the hybrid (Rust + Python-fallback) semantics in that mode.

This module does the registration eagerly at import time. Callers should
ensure `import orbit_wars_rust` happens before the first
`make("orbit_wars", ...)` call.

`_patches.apply()` is currently a no-op (kept for forward compatibility
with future hot-path monkey patches; see `_patches.py` for details).
"""

from __future__ import annotations

from kaggle_environments import register

from . import _facade, _patches

_patches.apply()
register("orbit_wars", _facade.environment_dict())

__all__ = ["_facade", "_patches"]
