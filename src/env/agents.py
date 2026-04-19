"""Agent registry and version resolution.

Maps short names to importable agent callables. Keeps the evaluation
framework decoupled from concrete agent modules.
"""

from __future__ import annotations

import importlib
import subprocess
from functools import lru_cache
from typing import Any, Callable

AgentCallable = Callable[..., Any]

AGENT_REGISTRY: dict[str, str] = {
    "baseline_v1": "pipeline.case1.baseline.agent:agent",
    "baseline_v2": "pipeline.case2.baseline.agent:agent",
    "case0": "pipeline.case0.main:agent",
    "case3_il_v1": "pipeline.case3.policy.agent:agent",
    "random": "random",
}


def list_agents() -> list[str]:
    return sorted(AGENT_REGISTRY.keys())


def resolve(name: str) -> AgentCallable | str:
    """Return a callable or a kaggle_environments built-in name string."""
    if name not in AGENT_REGISTRY:
        raise KeyError(f"unknown agent: {name!r} (known: {list_agents()})")
    target = AGENT_REGISTRY[name]
    if ":" not in target:
        return target
    module_path, attr = target.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)  # type: ignore[no-any-return]


@lru_cache(maxsize=1)
def agent_version(name: str | None = None) -> str:
    """Short git SHA of the working tree, or '' if git is unavailable."""
    del name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
