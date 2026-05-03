"""Backend selection facade.

`environment_dict()` returns the registry payload that
`kaggle_environments.register("orbit_wars", ...)` consumes. The keys are
identical to the upstream `kaggle_environments/envs/orbit_wars` registry —
we only swap out `interpreter`. Everything else (renderer, html_renderer,
specification, agents) is reused verbatim from the vendored Python copy.

## Default: backend = `python`

The default backend is the **upstream Python interpreter** (the vendored
copy of `kaggle_environments/envs/orbit_wars`). Importing this package
therefore preserves the exact behavior every existing call site already
sees — `make("orbit_wars", ...)` / `env.step` / `env.run` resolve to the
upstream Python implementation.

## Switching backends in code

Use the public helpers exported from `orbit_wars_rust`:

    import orbit_wars_rust
    orbit_wars_rust.use_rust()       # opt into the Rust interpreter
    orbit_wars_rust.use_python()     # back to the upstream Python (default)
    with orbit_wars_rust.backend("rust"):
        env.run([...])               # scoped switch with auto-restore

Or set the attribute directly:

    orbit_wars_rust.set_backend("rust")
    print(orbit_wars_rust.get_backend())   # → "rust"

The active interpreter is chosen lazily at each call, so flipping the
backend takes effect on the next `env.step` without re-importing.

## Backend = `rust` semantics (hybrid)

The Rust port does NOT implement initial planet generation
(`generate_planets`); we delegate the **generation-touching turns** back
to Python so the global `random` cursor stays in lock-step with upstream:

  1. The very first step (planets are still empty) — runs `generate_planets`
     in Python.
  2. Each step where `(obs.step + 1) in COMET_SPAWN_STEPS` (i.e. steps
     49 / 149 / 249 / 349 / 449 — the turn before each spawn) — pre-samples
     attempts in Python and dispatches to Rust `fast_generate_comet_paths`.

All other steps go through the Rust interpreter. This costs ~6 Python
steps per 500-turn episode (~1.2%) when the Rust backend is active and
preserves full upstream compatibility.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from typing import Any, Iterator, Literal

from orbit_wars_vendor import (
    agents,
    html_renderer,
    interpreter as python_interpreter,
    renderer,
    specification,
)
from orbit_wars_vendor.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    distance,
)

# Hoist the PyO3 extension import to module load time so the per-call
# `from ... import` cost (a dict lookup + binding) is paid once.
from orbit_wars_rust._lib import interpreter as _rust_step  # noqa: E402


def _rust_interpreter(state: Any, env: Any) -> Any:
    """Forward to the PyO3 extension module."""
    return _rust_step(state, env)


def _get_field(obs: Any, key: str) -> Any:
    """Read `obs.key` falling back to `obs[key]`. Returns None if neither path
    resolves — both shapes (SimpleNamespace and dict) appear in practice."""
    if obs is None:
        return None
    value = getattr(obs, key, None)
    if value is not None:
        return value
    try:
        return obs.get(key)
    except AttributeError:
        return None


def _observation(state: Any) -> Any:
    return getattr(state[0], "observation", None) or state[0].get("observation")


def _planets_present(state: Any) -> bool:
    """True iff `state[0].observation.planets` has been generated already."""
    return bool(_get_field(_observation(state), "planets"))


def _is_comet_spawn_turn(state: Any) -> bool:
    """True iff the upcoming interpreter call should spawn a new comet group.

    Mirrors the upstream check `if (step + 1) in COMET_SPAWN_STEPS:` —
    `obs.step` is the turn index *before* the interpreter runs. So when the
    framework presents `obs.step == 49`, this turn is the one that spawns the
    step-50 comet group, etc.
    """
    obs = _observation(state)
    step = _get_field(obs, "step")
    if step is None:
        return False
    try:
        return (int(step) + 1) in COMET_SPAWN_STEPS
    except (TypeError, ValueError):
        return False


def _expire_comets_at_path_end(state: Any) -> None:
    """Mirror upstream's pre-launch comet expiration block (orbit_wars.py:419-439).

    A comet whose `path_index` has reached the end of its path is removed
    from `planets`, `initial_planets`, `comet_planet_ids`, and the parent
    comet group. The Rust interpreter does the same expiration AFTER comet
    advance; doing it here BEFORE the spawn check keeps `comet_planet_ids`
    consistent with what upstream sees at the spawn-decision moment.
    """
    obs = _observation(state)
    comets = list(_get_field(obs, "comets") or [])
    if not comets:
        return
    expired: set[int] = set()
    for group in comets:
        idx = group["path_index"] if isinstance(group, dict) else group.path_index
        planet_ids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
        paths = group["paths"] if isinstance(group, dict) else group.paths
        for i, pid in enumerate(planet_ids):
            if idx >= len(paths[i]):
                expired.add(pid)
    if not expired:
        return
    planets = [p for p in (_get_field(obs, "planets") or []) if int(p[0]) not in expired]
    initial_planets = [
        p for p in (_get_field(obs, "initial_planets") or []) if int(p[0]) not in expired
    ]
    comet_planet_ids = [
        pid for pid in (_get_field(obs, "comet_planet_ids") or []) if int(pid) not in expired
    ]
    new_comets = []
    for group in comets:
        if isinstance(group, dict):
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired]
            if group["planet_ids"]:
                new_comets.append(group)
        else:
            kept = [pid for pid in group.planet_ids if pid not in expired]
            if kept:
                group.planet_ids = kept
                new_comets.append(group)

    for agent_state in state:
        agent_obs = getattr(agent_state, "observation", None) or agent_state.get("observation")
        if agent_obs is None:
            continue
        for key, value in (
            ("planets", planets),
            ("initial_planets", initial_planets),
            ("comet_planet_ids", comet_planet_ids),
            ("comets", new_comets),
        ):
            try:
                setattr(agent_obs, key, value)
            except (AttributeError, TypeError):
                pass
            try:
                agent_obs[key] = value
            except TypeError:
                pass


def _spawn_comet_via_rust(state: Any, env: Any) -> bool:
    """Spawn a new comet group using the Rust path generator.

    Mirrors the upstream interpreter's `if (step + 1) in COMET_SPAWN_STEPS`
    block: pre-samples up to 300 `(e, a, phi)` triples from `random.uniform`
    in the same order as upstream, then calls `_lib.fast_generate_comet_paths`
    to find the first surviving orbit and appends 4 new comet planets.

    Returns True if a comet group was spawned (matching upstream success
    path), False otherwise.
    """
    from orbit_wars_rust._lib import fast_generate_comet_paths

    # Expire path-ended comets first (matches upstream order).
    _expire_comets_at_path_end(state)

    obs = _observation(state)
    step = int(_get_field(obs, "step") or 0)
    spawn_step = step + 1
    if spawn_step not in COMET_SPAWN_STEPS:
        return False

    angular_velocity = float(_get_field(obs, "angular_velocity") or 0.0)
    comet_speed = float(env.configuration.cometSpeed)
    initial_planets = list(_get_field(obs, "initial_planets") or [])
    comet_planet_ids = list(_get_field(obs, "comet_planet_ids") or [])
    planets = list(_get_field(obs, "planets") or [])

    # Draw 300 (e, a, phi) triples in upstream order. Upstream's loop is
    #   for _ in range(300):
    #       e = random.uniform(0.75, 0.93)
    #       a = random.uniform(60, 150)
    #       perihelion check (continue does NOT consume more random)
    #       phi = random.uniform(pi/6, pi/3)
    # When perihelion < threshold, upstream `continue`s before drawing phi.
    # We replicate that exactly so the global random cursor lands where
    # upstream would put it.
    attempts: list[tuple[float, float, float]] = []
    for _ in range(300):
        e = random.uniform(0.75, 0.93)
        a = random.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            attempts.append((e, a, 0.0))  # phi is unused; Rust filters by perihelion too
            continue
        phi = random.uniform(math.pi / 6, math.pi / 3)
        attempts.append((e, a, phi))

    paths = fast_generate_comet_paths(
        attempts,
        spawn_step,
        angular_velocity,
        comet_speed,
        initial_planets,
        comet_planet_ids,
    )
    if paths is None:
        return False

    # Mirror upstream: ship count = min of 4 random.randint(1, 99)
    comet_ships = min(
        random.randint(1, 99),
        random.randint(1, 99),
        random.randint(1, 99),
        random.randint(1, 99),
    )

    # Append the new comet group + 4 planets, mutating the same lists the
    # framework holds. The next Rust step will advance them.
    next_id = max(int(p[0]) for p in planets) + 1
    new_group = {"planet_ids": [], "paths": paths, "path_index": -1}
    for i, p_path in enumerate(paths):
        pid = next_id + i
        new_group["planet_ids"].append(pid)
        comet_planet_ids.append(pid)
        new_planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
        planets.append(new_planet)
        # initial_planets gets a copy too (upstream uses planet[:])
        initial_planets.append(list(new_planet))

    comets = list(_get_field(obs, "comets") or [])
    comets.append(new_group)

    # Push the mutated lists back onto every agent's observation.
    for agent_state in state:
        agent_obs = getattr(agent_state, "observation", None) or agent_state.get("observation")
        if agent_obs is None:
            continue
        # Both attribute and item assignment to keep Struct.__dict__ in sync.
        for key, value in (
            ("planets", planets),
            ("initial_planets", initial_planets),
            ("comet_planet_ids", comet_planet_ids),
            ("comets", comets),
        ):
            try:
                setattr(agent_obs, key, value)
            except (AttributeError, TypeError):
                pass
            try:
                agent_obs[key] = value
            except TypeError:
                pass
    return True


# --- Backend selection state ---------------------------------------------
#
# Module-level state read on every interpreter call so flipping it takes
# effect on the next `env.step`. Default is the upstream Python interpreter
# so existing call sites preserve their pre-existing behavior.

Backend = Literal["python", "rust"]

_VALID_BACKENDS: tuple[Backend, ...] = ("python", "rust")

_active_backend: Backend = "python"
_rust_comet_enabled: bool = True


def get_backend() -> Backend:
    """Return the currently active backend (`"python"` or `"rust"`)."""
    return _active_backend


def set_backend(backend: Backend) -> None:
    """Switch the active backend. Takes effect on the next `env.step`.

    Raises `ValueError` for unknown backends so typos surface immediately
    rather than silently routing through the python fallback.
    """
    global _active_backend
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_VALID_BACKENDS}"
        )
    _active_backend = backend


def use_python() -> None:
    """Convenience wrapper: `set_backend("python")`."""
    set_backend("python")


def use_rust() -> None:
    """Convenience wrapper: `set_backend("rust")`."""
    set_backend("rust")


@contextmanager
def backend(backend_name: Backend) -> Iterator[None]:
    """Scoped backend switch: ``with backend("rust"): env.run(...)``.

    Restores the previous backend on exit even if the body raises.
    """
    prev = get_backend()
    set_backend(backend_name)
    try:
        yield
    finally:
        set_backend(prev)


def set_rust_comet(enabled: bool) -> None:
    """Toggle the Rust-accelerated comet spawn (only active under
    `backend == "rust"`). Default: enabled.

    Set to False to delegate comet spawn turns back to the upstream
    Python `generate_comet_paths` for strict bit-exact comparisons.
    """
    global _rust_comet_enabled
    _rust_comet_enabled = bool(enabled)


def get_rust_comet() -> bool:
    """Return whether Rust comet spawn is enabled."""
    return _rust_comet_enabled


def interpreter(state: Any, env: Any) -> Any:
    # Backend selection. Default is `python` — the upstream
    # `kaggle_environments` interpreter — so that every existing call site
    # behaves exactly as before importing this module. Use
    # `orbit_wars_rust.use_rust()` (or `set_backend("rust")` /
    # `with orbit_wars_rust.backend("rust"): ...`) to opt into the Rust path.
    if _active_backend != "rust":
        return python_interpreter(state, env)

    # Bootstrap (planets empty): upstream runs `generate_planets` and the
    # whole homeworld-assignment block. We delegate to Python because:
    #   1. generate_planets is a rejection loop bound to Python's global
    #      RNG cursor — breaking that breaks parity for downstream draws.
    #   2. It runs once per episode (~8-15 ms) so the speedup ceiling is
    #      <1% of per-episode wall-clock.
    if not _planets_present(state):
        return python_interpreter(state, env)

    # Comet spawn turns: spawn via Rust, then hand off to Rust interpreter.
    if _is_comet_spawn_turn(state):
        if _rust_comet_enabled:
            _spawn_comet_via_rust(state, env)
            return _rust_interpreter(state, env)
        return python_interpreter(state, env)

    return _rust_interpreter(state, env)


def environment_dict() -> dict[str, Any]:
    return {
        "interpreter": interpreter,
        "renderer": renderer,
        "html_renderer": html_renderer,
        "specification": specification,
        "agents": agents,
    }
