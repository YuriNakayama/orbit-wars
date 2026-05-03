"""baseline_v5 — port of Kaggle notebook `orbit-star-wars-lb-max-1224`.

Adapted from https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224
by Roman Tamrazov (Apache License 2.0).

Module layout:
- `core/config.py`         — tunable constants
- `core/types.py`          — Planet / Fleet / ShotOption / Mission
- `core/physics.py`        — pure geometry / sun-safe routing
- `core/world_helpers.py`  — arrival ledger / timeline simulation / FFA detection
- `agent_full.py`          — WorldModel, policy/score/plan, agent entrypoint
- `agent.py`               — public re-exports
"""

from .agent import agent, build_world

__all__ = ["agent", "build_world"]
