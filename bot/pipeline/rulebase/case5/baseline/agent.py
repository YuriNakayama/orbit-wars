"""Public agent entrypoint for baseline_v5.

Re-exports the WorldModel + strategy from `agent_full.py`. Pure helpers
(config, types, physics, world helpers) live in `core/`.

`agent_full.py` is a verbatim port of the LB1224 Kaggle notebook. It is held
back from the planner-style decomposition applied to case1/case4 because
upstream-parity tracing matters more than internal structure here. See the
case5 README for the rationale.
"""

from __future__ import annotations

from .agent_full import agent, build_world

__all__ = ["agent", "build_world"]
