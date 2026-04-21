"""Public agent entrypoint for baseline_v5.

Re-exports the WorldModel + strategy from `agent_full.py`. Pure helpers
(config, types, physics, world helpers) live in `core/`.
"""

from __future__ import annotations

from .agent_full import agent, build_world

__all__ = ["agent", "build_world"]
