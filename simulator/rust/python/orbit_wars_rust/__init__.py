"""Rust-backed Orbit Wars simulator.

Importing this package registers `orbit_wars` with `kaggle_environments` so
that `kaggle_environments.make("orbit_wars", ...)` dispatches to either the
Rust interpreter (default) or the Apache-2.0 vendored Python interpreter,
controlled by the `ORBIT_WARS_BACKEND` environment variable.

This module does the registration eagerly at import time. Callers must
ensure `import orbit_wars_rust` happens before the first `make("orbit_wars",
...)` call. The recommended location is `bot/src/__init__.py` so that any
module importing from `bot.src.*` triggers the registration.

Side effect: this import also installs hot-path monkey patches against
`kaggle_environments.utils.{structify, process_schema}` to remove
avoidable per-step overhead. Set `ORBIT_WARS_FAST_PATCH=0` to disable
them for a clean A/B benchmark. See `_patches.py` for details.
"""

from __future__ import annotations

from kaggle_environments import register

from . import _facade, _patches

_patches.apply()
register("orbit_wars", _facade.environment_dict())

__all__ = ["_facade", "_patches"]
