"""Rust-backed Orbit Wars simulator.

Importing this package registers `orbit_wars` with `kaggle_environments` so
that `kaggle_environments.make("orbit_wars", ...)` keeps working unchanged
(default backend is the upstream Python interpreter).

## Recommended entry point: `run`

A single function covers every speed tier — pick by argument:

    import orbit_wars_rust

    # 1 match
    result = orbit_wars_rust.run(["random", "random"], seed=0)

    # N matches in-process
    results = orbit_wars_rust.run(["random", "random"], seeds=range(30))

    # N matches across worker processes
    results = orbit_wars_rust.run(
        ["random", "random"], seeds=range(30),
        parallel=8, mp_context="fork",
    )

Each result is a small `{"seed", "turns", "rewards", "statuses"}` dict
suitable for IPC. Default `mp_context="spawn"` is safe with PyTorch /
CUDA; switch to ``"fork"`` for pure-Python agents (3-5× faster startup).

## Lower-level alternatives

When the caller needs the full `env.steps` history (replay JSON,
per-frame inspection), build the env yourself and use one of:

    env = make("orbit_wars", configuration={"agents": 2})

    orbit_wars_rust.run_episode(env, agents)             # 1 match, full steps
    orbit_wars_rust.run_episodes(env, agents, seeds)     # reuse env across seeds
    orbit_wars_rust.run_episodes_parallel(...)           # raw multi-process helper

Backend selection (only needed when not using `run`):

    orbit_wars_rust.use_rust()       # default for new code
    orbit_wars_rust.use_python()     # fall back to upstream
    with orbit_wars_rust.backend("rust"): env.run([...])  # scoped switch

The active backend is read on every `env.step`; flipping it takes effect
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
    run,
    run_episode,
    run_episodes,
    run_episodes_parallel,
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
    "run",
    "run_episode",
    "run_episodes",
    "run_episodes_parallel",
    "set_backend",
    "set_rust_comet",
    "use_python",
    "use_rust",
]
