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
    "baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent",
    "baseline_v2": "pipeline.rulebase.case2.baseline.agent:agent",
    "baseline_v3": "pipeline.rulebase.case3.baseline.agent:agent",
    "baseline_v4": "pipeline.rulebase.case4.baseline.agent:agent",
    "baseline_v5": "pipeline.rulebase.case5.baseline.agent:agent",
    "baseline_v6": "pipeline.rulebase.case6.baseline.agent:agent",
    "baseline_v7": "pipeline.rulebase.case7.baseline.agent:agent",
    "baseline_v8": "pipeline.rulebase.case8.baseline.agent:agent",
    "baseline_v9": "pipeline.rulebase.case9.baseline.agent:agent",
    "jax_v4": "pipeline.rulebase.case4.baseline_jax.agent_jax:agent",
    "jax_v8": "pipeline.rulebase.case8.baseline_jax.agent_jax:agent",
    "case0": "pipeline.rulebase.case0.main:agent",
    "il_v0": "pipeline.imitation.case0.policy.agent:agent",
    "il_v1": "pipeline.imitation.case1.policy.agent:agent",
    "il_v2": "pipeline.imitation.case2.policy.agent:agent",
    "il_v2_phase1": "pipeline.imitation.case2.policy.agent_phase1:agent",
    "il_v3": "pipeline.imitation.case3.policy.agent_phase2:agent",
    "il_v4": "pipeline.imitation.case4.policy.agent:agent",
    "il_v5": "pipeline.imitation.case5.policy.agent:agent",
    "il_v6": "pipeline.imitation.case6.policy.agent:agent",
    "il_v7": "pipeline.imitation.case7.policy.agent:agent",
    "il_v8": "pipeline.imitation.case8.policy.agent:agent",
    "il_v9_three_head": "pipeline.imitation.case9.policy.agent:agent",
    "il_v9_candidate": "pipeline.imitation.case9.policy.agent:agent",
    "il_v9_candidate_ships": "pipeline.imitation.case9.policy.agent:agent",
    "il_v9_template_ships": "pipeline.imitation.case9.policy.agent:agent",
    "il_v9_dual": "pipeline.imitation.case9.policy.agent:agent",
    "il_v9_per_planet": "pipeline.imitation.case9.policy.agent:agent",
    "il_v10_candidate": "pipeline.imitation.case10.policy.agent:agent",
    "il_v10_template": "pipeline.imitation.case10.policy.agent:agent",
    "il_v11_per_planet": "pipeline.imitation.case11.policy.agent:agent",
    "rl_v0": "pipeline.reinforce.case0.policy.agent:agent",
    "rl_v1": "pipeline.reinforce.case1.policy.agent:agent",
    "rl_v3": "pipeline.reinforce.case3.policy.agent:agent",
    "rl_v4": "pipeline.reinforce.case4.policy.agent:agent",
    "rl_v5": "pipeline.reinforce.case5.policy.agent:agent",
    "rl_v6": "pipeline.reinforce.case6.policy.agent:agent",
    "rl_v7": "pipeline.reinforce.case7.policy.agent:agent",
    "random": "random",
}


def list_agents() -> list[str]:
    return sorted(AGENT_REGISTRY.keys())


def resolve(name: str) -> AgentCallable | str:
    """Return a callable or a the local simulator built-in name string."""
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
