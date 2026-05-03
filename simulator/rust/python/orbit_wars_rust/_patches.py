"""Hot-path monkey patches for `kaggle_environments` — DISABLED.

We attempted to remove avoidable Python overhead in `env.step` (332µs of
framework time per call, 96% of the total) by patching `structify` and
`process_schema`. Both attempts failed for fundamental reasons captured
here so we don't repeat the experiment without new evidence.

## Why `process_schema` cannot be patched safely

The upstream `process_schema` does:

    data = default_schema(schema, deepcopy(data))

`default_schema` walks the schema and inserts missing default keys into
`data` *in place*. The defaults include `observation.remainingOverageTime`
which `Agent.act` later subtracts from. If we replace the deepcopy with
a shallow copy, the default insertion sometimes fails to materialize on
the inner observation dict (because `default_schema` recurses and rebinds
the inner dict, but the outer caller still holds a reference to the
unchanged inner dict). Result: `AttributeError: 'Struct' object has no
attribute 'remainingOverageTime'` from the agent timing path.

## Why `structify` cannot be patched safely

`Struct` inherits from `dict`. Upstream `structify(dict)` deep-wraps every
nested dict so attribute access (`obs.angular_velocity`) and key access
(`obs["angular_velocity"]`) both work. Crucially, `Struct.__init__`
populates `self.__dict__` from the kwargs, but our Rust extension writes
back to the observation via `obj.set_item(key, value)` which only updates
the dict-storage and NOT the `__dict__`. The framework recovers from this
because every `env.step` re-runs `structify(...)` after the interpreter,
which reconstructs the `Struct` from scratch and re-syncs `__dict__`.

Any idempotent patch (skip recursion when input is already `Struct`)
breaks that recovery path. Concretely:

  1. Rust mutates `obs` via `set_item("planets", ...)` only.
  2. Patched `structify` short-circuits because `obs` is already `Struct`.
  3. `Agent.act` reads `obs.remainingOverageTime` (attribute access).
  4. AttributeError because `Struct.__dict__` was never refreshed.

Fixing this from the Rust side would require calling `setattr` on every
key — but that defeats the speed advantage of in-place mutation, since
PyO3's `setattr` goes through Python's descriptor protocol.

## What works instead

Because the framework overhead is structural (deep clones in service of
mutation safety), the only effective speed-up paths are:

  * **`run_episode` batch API** (new function, opt-in via env var) —
    runs the whole episode in Rust, only paying the framework cost once.
    See `docs/plans/rust-simulator/optimization-tradeoffs.md` Phase 2.
  * **multiprocessing self-play** (already implemented in
    `bot/src/dataset/selfplay/runner.py`) — orthogonal, scales to N cores.

This module is kept as a no-op so other code can `import orbit_wars_rust._patches`
without ImportError.
"""

from __future__ import annotations

from typing import Any, Callable

_originals: dict[str, Callable[..., Any]] = {}


def apply() -> None:
    """No-op (see module docstring)."""
    return


def revert() -> None:
    """No-op (no patches to revert)."""
    return
