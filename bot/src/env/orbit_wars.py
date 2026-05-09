"""Helpers for running Orbit Wars through the in-repo simulator backend."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from utils.repo_root import find_repo_root


def _ensure_simulator_path() -> None:
    repo_root = find_repo_root(Path(__file__))
    simulator_python = repo_root / "simulator" / "rust" / "python"
    path = str(simulator_python)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_simulator() -> Any:
    _ensure_simulator_path()
    try:
        import orbit_wars_rust
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "orbit_wars_rust is required for local simulation; "
            "build it with `(cd simulator/rust && uv run maturin develop --release)`."
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
    """Create an Orbit Wars env registered to the in-repo Rust simulator."""
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
    """Run one episode with the in-repo simulator implementation."""
    simulator = _load_simulator()
    return simulator.run_episode(env, list(agents))
