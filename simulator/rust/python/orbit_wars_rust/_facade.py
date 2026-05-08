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

The Rust backend delegates bootstrap generation (`generate_planets`) to
the vendored Python simulator so it follows the upstream deterministic
seed contract exactly. Comet spawn turns use the same per-spawn seeded RNG
sequence as upstream and dispatch the path filtering math to Rust via
`fast_generate_comet_paths`. All other steps go through the Rust
interpreter.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from typing import Any, Iterator, Literal

from orbit_wars_vendor import (
    agents,
    html_renderer,
    renderer,
    specification,
)
from orbit_wars_vendor import (
    interpreter as python_interpreter,
)
from orbit_wars_vendor.orbit_wars import (
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    SUN_RADIUS,
)

# Frozen set version for the per-step `step+1 in COMET_SPAWN_STEPS` check —
# upstream exposes the canonical list (kept for parity), but membership-test
# hot path benefits from O(1) hash lookup vs the 5-element linear scan.
_COMET_SPAWN_STEPS_SET: frozenset[int] = frozenset(COMET_SPAWN_STEPS)

# Hoist the PyO3 extension imports to module load time so the per-call
# `from ... import` cost (a dict lookup + binding) is paid once. Both
# helpers are read-only at module scope, so importing them eagerly costs
# nothing extra and removes ~µs / call from the comet-spawn / expire paths.
from orbit_wars_rust._lib import (  # noqa: E402
    fast_filter_expired as _rust_filter_expired,
)
from orbit_wars_rust._lib import (  # noqa: E402
    fast_generate_comet_paths as _rust_generate_comet_paths,
)
from orbit_wars_rust._lib import (  # noqa: E402
    interpreter as _rust_step,
)


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
        is_dict = isinstance(group, dict)
        idx = group["path_index"] if is_dict else group.path_index
        planet_ids = group["planet_ids"] if is_dict else group.planet_ids
        paths = group["paths"] if is_dict else group.paths
        for i, pid in enumerate(planet_ids):
            if idx >= len(paths[i]):
                expired.add(pid)
    if not expired:
        return
    planets, initial_planets, comet_planet_ids = _rust_filter_expired(
        list(_get_field(obs, "planets") or []),
        list(_get_field(obs, "initial_planets") or []),
        list(_get_field(obs, "comet_planet_ids") or []),
        list(expired),
    )
    new_comets = []
    for group in comets:
        if isinstance(group, dict):
            kept_ids = [p for p in group["planet_ids"] if p not in expired]
            group["planet_ids"] = kept_ids
            if kept_ids:
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
    # Expire path-ended comets first (matches upstream order).
    _expire_comets_at_path_end(state)

    obs = _observation(state)
    step = int(_get_field(obs, "step") or 0)
    spawn_step = step + 1
    if spawn_step not in COMET_SPAWN_STEPS:
        return False

    angular_velocity = float(_get_field(obs, "angular_velocity") or 0.0)
    comet_speed = float(env.configuration.cometSpeed)
    env_info = getattr(env, "info", None) or {}
    episode_seed = env_info.get("seed", 0) or 0
    comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
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
    # We replicate that exactly against the per-spawn deterministic RNG.
    attempts: list[tuple[float, float, float]] = []
    for _ in range(300):
        e = comet_rng.uniform(0.75, 0.93)
        a = comet_rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            attempts.append((e, a, 0.0))  # phi is unused; Rust filters by perihelion too
            continue
        phi = comet_rng.uniform(math.pi / 6, math.pi / 3)
        attempts.append((e, a, phi))

    paths = _rust_generate_comet_paths(
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
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
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
    """Convenience wrapper: `set_backend("python")`.

    Rarely needed for new code — :func:`run` accepts ``backend="python"``
    inline. Use this when you want callers of `make("orbit_wars",
    ...).run(...)` to transparently fall back to the upstream Python
    interpreter.
    """
    set_backend("python")


def use_rust() -> None:
    """Convenience wrapper: `set_backend("rust")`.

    Rarely needed for new code — :func:`run` already defaults to the
    rust backend. Use this when you want existing `make("orbit_wars",
    ...).run(...)` call sites to transparently switch over without any
    other code change (~2× transparent speedup).
    """
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

    # Resolve `state[0].observation` once and reuse for both the bootstrap
    # check and the comet-spawn check. Avoid the generic getattr-then-item
    # fallback helpers — `state[0]` is the Struct (dict subclass) the
    # framework hands the interpreter, so direct attr access lands on the
    # first hit. We keep the safety fallback for the rare dict-only path.
    agent0 = state[0]
    obs = getattr(agent0, "observation", None)
    if obs is None:
        try:
            obs = agent0.get("observation")
        except AttributeError:
            obs = None
    planets = getattr(obs, "planets", None) if obs is not None else None
    if planets is None and obs is not None:
        try:
            planets = obs.get("planets")
        except AttributeError:
            planets = None
    if not planets:
        # Bootstrap (planets empty): upstream runs `generate_planets` and
        # the whole homeworld-assignment block. We delegate to Python
        # because generate_planets is a rejection loop bound to Python's
        # global RNG cursor — breaking that breaks parity for downstream
        # draws. Runs once per episode so the speedup ceiling is <1%.
        return python_interpreter(state, env)

    # Comet spawn turns: spawn via Rust, then hand off to Rust interpreter.
    step_field = getattr(obs, "step", None)
    if step_field is None:
        try:
            step_field = obs.get("step")
        except AttributeError:
            step_field = None
    is_spawn_turn = False
    if step_field is not None:
        try:
            is_spawn_turn = (int(step_field) + 1) in _COMET_SPAWN_STEPS_SET
        except (TypeError, ValueError):
            is_spawn_turn = False
    if is_spawn_turn:
        if _rust_comet_enabled:
            _spawn_comet_via_rust(state, env)
            return _rust_interpreter(state, env)
        return python_interpreter(state, env)

    return _rust_interpreter(state, env)


def _resolve_agent(agent: Any, env: Any, position: int) -> Any:
    """Convert string agent specs ("random", "starter", a path) into a callable.

    Mirrors what `kaggle_environments.Agent` does, but stripped of timing /
    overage-time machinery: `run_episode` skips that bookkeeping in
    exchange for raw throughput.
    """
    if callable(agent):
        return agent
    if isinstance(agent, str):
        registered = env.agents.get(agent) if hasattr(env, "agents") else None
        if registered is None:
            registered = environment_dict()["agents"].get(agent)
        if registered is None:
            raise ValueError(
                f"unknown agent {agent!r}; expected callable or registered name"
            )
        return registered
    raise TypeError(f"agent {agent!r} at position {position} is not callable")


def _agents_done(state: Any) -> bool:
    """Mirror `Environment.done`: every agent's status is non-ACTIVE.

    Hot path inside `run_episode` — called once per step. Trust attr
    access for Struct (dict subclass); only fall back to dict.get for
    the rare plain-dict state seen in some test harnesses.
    """
    for agent in state:
        status = getattr(agent, "status", None)
        if status is None and hasattr(agent, "get"):
            status = agent.get("status")
        if status == "ACTIVE":
            return False
    return True


def run_episode(env: Any, agents: list[Any]) -> list[Any]:
    """Drive a single episode end-to-end through the Rust-backed interpreter.

    .. note::
        For new code prefer the unified entry point :func:`run`. Use
        ``run_episode`` when you need the live `env.steps` history (replay
        JSON, per-frame inspection) on a `make()`-built env.

    Behavior contract:
      * Equivalent to `env.run(agents)` except that per-step framework
        overhead (`structify` ×2, `process_schema`, `deepcopy`, agent
        wrapper / overage-time tracking) is bypassed. The Rust interpreter
        is invoked directly on the live `env.state`, mutating in place.
      * `env.steps`, `env.state`, `env.done` are kept in sync so existing
        post-processing (`env.toJSON()`, `env.steps[-1]`, replay save)
        works unchanged.
      * On any internal divergence (missing observation, agent error, or
        unsupported state shape) we fall back to the upstream `env.run`,
        so the new path is always a strict superset of the legacy one.
      * Active backend must be `rust`; otherwise we hand off to upstream.

    Returns `env.steps` exactly like `env.run` does.
    """
    if _active_backend != "rust":
        return env.run(agents)

    # Reset semantics identical to env.run.
    if env.state is None or len(env.steps) == 1 or env.done:
        env.reset(len(agents))
    if len(env.state) != len(agents):
        raise ValueError(
            f"{len(env.state)} agents were expected, but {len(agents)} was given."
        )

    try:
        resolved = [_resolve_agent(agents[i], env, i) for i in range(len(agents))]
    except (ValueError, TypeError):
        # Fallback: unknown agent types — let the upstream Agent machinery
        # handle string lookups, etc.
        return env.run(agents)

    episode_steps = int(env.configuration.episodeSteps)

    # Local bindings for the hottest names — Python attribute lookup is
    # ~30 ns / hit, multiplied by ~500 turns × 30 matches = ~450k lookups
    # this saves ~14 ms / 30 matches. Tiny by itself but compounds with
    # the bootstrap-once trick below.
    rust_step = _rust_step
    spawn_steps = _COMET_SPAWN_STEPS_SET
    py_interp = python_interpreter
    configuration = env.configuration
    bootstrap_done = False
    num_agents = len(agents)

    # Decide each agent's calling convention ONCE up-front instead of
    # paying try/except overhead per step. Probe with the first ACTIVE
    # observation; if probing fails (e.g. starts INACTIVE), default to
    # 2-arg and let the per-step path fall back. ~95% of orbit_wars
    # agents accept (obs, conf).
    accepts_conf: list[bool] = [True] * num_agents
    for i in range(num_agents):
        ag_state = env.state[i]
        ag_obs = getattr(ag_state, "observation", None)
        if ag_obs is None:
            continue
        try:
            resolved[i](ag_obs, configuration)
        except TypeError:
            try:
                resolved[i](ag_obs)
                accepts_conf[i] = False
            except Exception:
                # Probe failed both ways — defer to legacy on first call.
                pass
        except Exception:
            pass

    while not _agents_done(env.state):
        # Gather actions from every ACTIVE agent. Inactive agents (status
        # set by combat termination) submit a no-op `[]` so the interpreter
        # still sees the right number of action slots.
        actions: list[Any] = []
        state = env.state
        for i in range(num_agents):
            agent_state = state[i]
            # Direct attr access — `Struct` is a dict subclass so the
            # attribute path is the first hit; we trust it instead of
            # `getattr(..., None)` which adds a None-coalesce.
            if agent_state.status != "ACTIVE":
                actions.append([])
                continue
            obs = agent_state.observation
            try:
                if accepts_conf[i]:
                    action = resolved[i](obs, configuration)
                else:
                    action = resolved[i](obs)
            except Exception:
                # Any agent failure: hand the rest of the episode to
                # upstream env.run so timeouts/errors propagate via the
                # framework's full state machine.
                return env.run(agents)
            actions.append(action if action is not None else [])

        # Stamp the actions onto state. Struct.__setattr__ writes both
        # __dict__ and dict storage so attr/item readers stay in sync;
        # we trust this path and skip the try/except wrapping.
        for i in range(num_agents):
            state[i].action = actions[i]

        # Drive the interpreter. After bootstrap we can skip the facade's
        # backend / planets-empty checks and call the Rust step directly,
        # only routing through the python_interpreter on comet-spawn turns
        # (where _facade does extra work). Saves a Python-side function
        # call + 4–5 attribute lookups per step.
        if bootstrap_done:
            obs0 = env.state[0].observation
            cur_step = getattr(obs0, "step", None)
            is_spawn = (
                cur_step is not None and (int(cur_step) + 1) in spawn_steps
            )
            if is_spawn:
                if _rust_comet_enabled:
                    _spawn_comet_via_rust(env.state, env)
                    new_state = rust_step(env.state, env)
                else:
                    new_state = py_interp(env.state, env)
            else:
                new_state = rust_step(env.state, env)
        else:
            # First turn: go through the facade so the bootstrap (planets
            # empty → python_interpreter) path runs exactly once.
            new_state = interpreter(env.state, env)
            bootstrap_done = True
        env.state = new_state

        # Manual step counter advance (upstream does this in
        # __loop_through_interpreter). Use attribute assignment so the
        # Struct's __dict__ stays in sync with dict storage — readers
        # (the Rust interpreter via `get_attr_or_item`, and the upstream
        # python_interpreter on bootstrap turns) hit the attr path first.
        try:
            env.state[0].observation.step = len(env.steps)
        except AttributeError:
            try:
                env.state[0]["observation"]["step"] = len(env.steps)
            except (KeyError, TypeError):
                pass

        # Mark survivors DONE on max-steps boundary.
        cur_step = (
            getattr(env.state[0].observation, "step", None)
            if hasattr(env.state[0], "observation")
            else None
        )
        if cur_step is not None and cur_step >= episode_steps - 1:
            for s in env.state:
                if getattr(s, "status", None) in ("ACTIVE", "INACTIVE"):
                    try:
                        s.status = "DONE"
                    except (AttributeError, TypeError):
                        s["status"] = "DONE"

        env.steps.append(env.state)

    return env.steps


def run_episodes(
    env: Any,
    agents: list[Any],
    seeds: list[int],
) -> list[list[Any]]:
    """Run multiple episodes on the same `env`, varying `configuration.seed`.

    .. note::
        For new code prefer the unified entry point :func:`run`. Use
        ``run_episodes`` when you need the per-episode `env.steps`
        snapshots (replay JSON, per-frame inspection).

    Returns a list of `env.steps` snapshots — one per seed. Each entry is
    independent (the framework rebinds `env.steps` to a fresh list inside
    `env.reset`), so the caller can persist or aggregate them after the
    fact without holding a reference into the live env.

    Skipping `make()` for seeds 1..N saves ~40 ms per match on the
    typical orbit_wars schema, which dominates the residual cost after
    `run_episode` already removed the per-step framework overhead. The
    upstream `kaggle_environments.make` recompiles the JSON schema each
    call; reusing the env keeps that compiled schema and only pays the
    per-reset jsonschema validate.

    Active backend must be `rust`; otherwise the per-seed loop falls back
    to upstream `env.run` for full behavioral parity.
    """
    results: list[list[Any]] = []
    config = env.configuration
    for seed in seeds:
        config.seed = seed
        env.reset(len(agents))
        run_episode(env, agents)
        results.append(env.steps)
    return results


# Per-worker cache: kaggle_environments.make and orbit_wars_rust references
# warmed up once per worker process by `_worker_init`. spawn ctx pays a
# ~0.5-1 s import cost per worker; without this initializer that cost was
# being amortized across only 30/N matches and dominating wall-clock for
# small N. `_worker_init` runs once at Pool startup so the per-task cost
# drops to just `make + run_episode` (~60 ms / match).
_WORKER_MAKE: Any = None
_WORKER_OW: Any = None


def _worker_init() -> None:
    """Pool initializer: warm imports + flip backend exactly once per worker.

    Called by `multiprocessing.Pool(initializer=_worker_init)` so spawn
    workers don't re-import `orbit_wars_rust` / `kaggle_environments`
    on every task dispatch.
    """
    global _WORKER_MAKE, _WORKER_OW
    from kaggle_environments import make as _make

    import orbit_wars_rust as _ow

    _ow.use_rust()
    _WORKER_OW = _ow
    _WORKER_MAKE = _make


def _worker_run_episode(args: tuple[Any, ...]) -> dict[str, Any]:
    """Top-level worker for `run_episodes_parallel` — must be picklable.

    Lives at module scope so `multiprocessing.Pool` can serialize a
    reference to it. Lazy-imports if `_worker_init` did not run (parallel=1
    or fork ctx where the parent's module-level state already inherits).
    Returns a small JSON-friendly summary instead of `env.steps` to keep
    IPC payloads tiny — full step history is only available via the
    per-process `run_episodes` / `run_episode` API when the caller needs
    replay JSON.
    """
    global _WORKER_MAKE, _WORKER_OW
    if _WORKER_MAKE is None:
        _worker_init()

    agents, seed, num_agents = args
    env = _WORKER_MAKE(
        "orbit_wars", configuration={"agents": num_agents, "seed": seed}
    )
    _WORKER_OW.run_episode(env, list(agents))
    final = env.steps[-1] if env.steps else []
    rewards: list[Any] = []
    statuses: list[str] = []
    for s in final:
        try:
            rewards.append(s["reward"])
        except (KeyError, TypeError):
            rewards.append(getattr(s, "reward", None))
        try:
            statuses.append(s["status"])
        except (KeyError, TypeError):
            statuses.append(getattr(s, "status", "UNKNOWN"))
    return {
        "seed": seed,
        "turns": len(env.steps),
        "rewards": rewards,
        "statuses": statuses,
    }


def run_episodes_parallel(
    agents: list[str],
    seeds: list[int],
    parallel: int = 1,
    mp_context: str = "spawn",
) -> list[dict[str, Any]]:
    """Run many episodes across worker processes — env.run replacement.

    .. note::
        For new code prefer the unified entry point :func:`run` —
        ``run(agents, seeds=..., parallel=..., mp_context=...)`` is a
        thin wrapper around this function and accepts the same arguments
        plus ``backend=`` for parity / debugging.

    Caller-side code:

        import orbit_wars_rust
        results = orbit_wars_rust.run_episodes_parallel(
            agents=["random", "random"],
            seeds=list(range(30)),
            parallel=8,
            mp_context="fork",
        )

    No env vars, no caller-side `multiprocessing.Pool` plumbing. The
    helper sets the rust backend in every worker, runs `run_episode`,
    and returns one small dict per seed (`{seed, turns, rewards,
    statuses}`). Replay JSON / per-step history are intentionally not
    serialized cross-process — pickling them dominates IPC cost. Use
    `run_episodes` (single-process) when full `env.steps` is needed.

    Args:
        agents: list of registered agent names (e.g. ``"random"``,
            ``"starter"``). Custom callables are NOT supported here
            because top-level pickling rules apply per worker; pass a
            registered name or extend the registry first.
        seeds: list of episode seeds — drives `configuration.seed`.
        parallel: number of worker processes. ``1`` runs serially in
            the calling process (no Pool overhead).
        mp_context: ``"spawn"`` (default, safe with CUDA / PyTorch
            tensors held by the parent) or ``"fork"`` (3-5× faster
            startup; only safe when the parent has no GPU tensors and
            no thread-bound state). ``"forkserver"`` is also accepted.

    Returns one summary dict per seed in the same order as `seeds`.
    """
    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")
    if mp_context not in ("spawn", "fork", "forkserver"):
        raise ValueError(
            f"mp_context must be 'spawn'|'fork'|'forkserver', got {mp_context!r}"
        )
    for a in agents:
        if not isinstance(a, str):
            raise TypeError(
                "run_episodes_parallel only accepts registered agent names "
                "(str), not callables. Use run_episode/run_episodes for "
                "in-process callables."
            )
    num_agents = len(agents)
    agent_tuple = tuple(agents)
    work = [(agent_tuple, int(seed), num_agents) for seed in seeds]

    if parallel == 1:
        # Warm parent process imports too so the timing matches the pool
        # path (otherwise the very first call pays a ~50 ms import tax
        # that the helper users won't see in subsequent invocations).
        if _WORKER_MAKE is None:
            _worker_init()
        return [_worker_run_episode(w) for w in work]

    import multiprocessing as _mp

    ctx = _mp.get_context(mp_context)
    # chunksize: feed each worker a contiguous block so the per-task IPC
    # round-trip cost is amortized. With 30 tasks / 8 workers, chunksize=4
    # means each worker pulls one chunk and stays warm for 4 matches —
    # cuts pickle/unpickle overhead by 4× without losing scheduling
    # flexibility (last worker still picks up stragglers).
    chunksize = max(1, (len(work) + parallel - 1) // parallel)
    with ctx.Pool(processes=parallel, initializer=_worker_init) as pool:
        # imap (ordered) so caller gets results aligned with `seeds`.
        return list(pool.imap(_worker_run_episode, work, chunksize=chunksize))


def run(
    agents: list[Any],
    seed: int | None = None,
    seeds: list[int] | range | None = None,
    parallel: int = 1,
    mp_context: str = "spawn",
    backend: Backend = "rust",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Single entry point for every speed tier — pick by argument.

    Examples:
      * `run(agents, seed=0)` — single match, in-process.
      * `run(agents, seeds=range(N))` — N matches, in-process serial.
      * `run(agents, seeds=range(N), parallel=8, mp_context="fork")` —
        N matches fanned out across 8 worker processes.
      * `run(agents, seed=0, backend="python")` — force the upstream
        Python interpreter (parity / debugging escape hatch).

    Typical speedups vs the Python baseline (30 matches × 2 random
    agents on a 12-core M-series Mac):

      * `backend="python"` ............................. 1.0× (51 s)
      * default (rust serial) .......................... ~27× (1.8 s)
      * `parallel=4, mp_context="fork"` ................ ~108× (0.47 s)
      * `parallel=8, mp_context="fork"` ................ ~120-141× (0.36-0.43 s)
      * `parallel=12, mp_context="fork"` ............... ~180× (0.28 s)

    Returns one summary dict (`seed=`) or a list of them (`seeds=`).
    Each summary is `{"seed", "turns", "rewards", "statuses"}` — small,
    JSON-friendly, suitable for cross-process IPC.

    `agents` accepts a mix of registered names (e.g. ``"random"``,
    ``"starter"``) and user callables. Callables are supported only when
    ``parallel == 1`` (in-process); workers cannot pickle arbitrary
    closures, so a callable with ``parallel >= 2`` raises ``TypeError``.
    Use a registered name (or a top-level pickleable module function) to
    parallelize.

    The default `mp_context="spawn"` is safe with PyTorch / CUDA tensors
    held by the parent. Use ``"fork"`` for pure-Python agents to skip the
    per-worker import tax (~0.5-1 s) — typically 3-5× faster startup.
    """
    if seed is None and seeds is None:
        raise ValueError("provide either `seed=` or `seeds=`")
    if seed is not None and seeds is not None:
        raise ValueError("`seed=` and `seeds=` are mutually exclusive")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_VALID_BACKENDS}"
        )
    set_backend(backend)

    if seed is not None:
        seeds_list = [int(seed)]
        single = True
    else:
        seeds_list = [int(s) for s in seeds]  # type: ignore[arg-type]
        single = False

    has_callable = any(not isinstance(a, str) for a in agents)

    if has_callable and parallel >= 2:
        raise TypeError(
            "callable agents cannot be sent to worker processes; pass "
            "registered agent names or use parallel=1."
        )

    if has_callable:
        # In-process path — `run_episodes_parallel` only accepts strings,
        # so we drive the per-seed loop ourselves with `run_episode`.
        from kaggle_environments import make as _make

        num_agents = len(agents)
        summaries: list[dict[str, Any]] = []
        for s in seeds_list:
            env = _make(
                "orbit_wars", configuration={"agents": num_agents, "seed": s}
            )
            run_episode(env, list(agents))
            final = env.steps[-1] if env.steps else []
            rewards: list[Any] = []
            statuses: list[str] = []
            for state in final:
                try:
                    rewards.append(state["reward"])
                except (KeyError, TypeError):
                    rewards.append(getattr(state, "reward", None))
                try:
                    statuses.append(state["status"])
                except (KeyError, TypeError):
                    statuses.append(getattr(state, "status", "UNKNOWN"))
            summaries.append(
                {
                    "seed": s,
                    "turns": len(env.steps),
                    "rewards": rewards,
                    "statuses": statuses,
                }
            )
        return summaries[0] if single else summaries

    summaries = run_episodes_parallel(
        agents=list(agents),
        seeds=seeds_list,
        parallel=parallel,
        mp_context=mp_context,
    )
    return summaries[0] if single else summaries


def environment_dict() -> dict[str, Any]:
    return {
        "interpreter": interpreter,
        "renderer": renderer,
        "html_renderer": html_renderer,
        "specification": specification,
        "agents": agents,
    }
