"""Helpers for running Orbit Wars through the in-repo simulator backend.

By default we load the Rust-backed `orbit_wars_rust` package (PyO3 native
extension). When `ORBIT_WARS_BACKEND=python` is set we instead register the
vendored upstream Python sim directly with `kaggle_environments` and skip the
Rust import — useful on hosts where the native wheel isn't available (e.g.
Kaggle Kernel without a pre-built manylinux wheel).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from utils.repo_root import find_repo_root

_PYTHON_BACKEND_ENV = "ORBIT_WARS_BACKEND"
_PYTHON_BACKEND_VALUE = "python"


def _want_python_backend() -> bool:
    return os.environ.get(_PYTHON_BACKEND_ENV, "").lower() == _PYTHON_BACKEND_VALUE


def _ensure_simulator_path() -> None:
    repo_root = find_repo_root(Path(__file__))
    paths = [
        repo_root / "simulator" / "rust" / "python",
        repo_root / "simulator" / "python",
    ]
    for path in paths:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


_PYTHON_BACKEND_REGISTERED = False


def _load_python_backend() -> Any:
    """Register vendored upstream Python sim with kaggle_environments."""
    global _PYTHON_BACKEND_REGISTERED
    _ensure_simulator_path()
    import orbit_wars_vendor
    from kaggle_environments import register

    if not _PYTHON_BACKEND_REGISTERED:
        register(
            "orbit_wars",
            {
                "interpreter": orbit_wars_vendor.interpreter,
                "renderer": orbit_wars_vendor.renderer,
                "html_renderer": orbit_wars_vendor.html_renderer,
                "specification": orbit_wars_vendor.specification,
                "agents": orbit_wars_vendor.agents,
            },
        )
        _PYTHON_BACKEND_REGISTERED = True
    return orbit_wars_vendor


def _load_simulator() -> Any:
    """Load the simulator backend selected by ORBIT_WARS_BACKEND.

    `python`  → register vendored upstream Python sim directly (no Rust needed)
    otherwise → load `orbit_wars_rust` (PyO3 native ext, default Rust backend)
    """
    _ensure_simulator_path()
    if _want_python_backend():
        return _load_python_backend()
    try:
        import orbit_wars_rust
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "orbit_wars_rust is required for local simulation; "
            "build it with `(cd simulator/rust && uv run maturin develop --release)`,"
            f" or set {_PYTHON_BACKEND_ENV}={_PYTHON_BACKEND_VALUE} to use the"
            " vendored Python sim."
        ) from exc
    orbit_wars_rust.use_rust()
    return orbit_wars_rust


def make_orbit_wars_env(
    *,
    seed: int | None = None,
    agents: int | None = None,
    episode_steps: int | None = None,
    debug: bool = False,
    configuration: dict[str, Any] | None = None,
    steps: list[Any] | None = None,
) -> Any:
    """Create an Orbit Wars env registered to the selected simulator backend."""
    _load_simulator()
    from kaggle_environments import make

    config: dict[str, Any] = dict(configuration or {})
    if seed is not None:
        config["seed"] = seed
    if agents is not None:
        config["agents"] = agents
    if episode_steps is not None:
        config["episodeSteps"] = episode_steps
    kwargs: dict[str, Any] = {"configuration": config, "debug": debug}
    if steps is not None:
        kwargs["steps"] = steps
    return make("orbit_wars", **kwargs)


def run_orbit_wars_episode(env: Any, agents: Sequence[Any]) -> list[Any]:
    """Run one episode with the selected simulator backend.

    Python backend has no `run_episode` helper (Rust-only optimization), so
    we fall back to a simple `env.run` loop.
    """
    simulator = _load_simulator()
    if _want_python_backend():
        env.run(list(agents))
        return cast(list[Any], env.steps)
    return cast(list[Any], simulator.run_episode(env, list(agents)))
