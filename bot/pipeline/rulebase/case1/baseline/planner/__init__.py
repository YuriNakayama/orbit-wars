"""Internal planner helpers for case1 strategy.

Decomposes the legacy 702-line `strategy.plan_moves` into orderable units:

- `option_collector.collect_capture_options_and_missions`
- `swarm_builder.build_swarm_missions`
- `mission_resolver.process_single_source_mission`
- `mission_resolver.process_multi_source_mission`
- `followup.emit_followup_moves`
- `evacuation.emit_evacuation_moves`
- `rear_guard.emit_rear_guard_moves`
- `finalizer.enforce_inventory_cap`

Each function is a pure-ish unit testable with synthetic WorldModel fixtures.
"""

from __future__ import annotations

from .evacuation import emit_evacuation_moves
from .finalizer import enforce_inventory_cap
from .followup import emit_followup_moves
from .mission_resolver import (
    process_multi_source_mission,
    process_single_source_mission,
)
from .option_collector import collect_capture_options_and_missions
from .rear_guard import emit_rear_guard_moves
from .swarm_builder import build_swarm_missions

__all__ = [
    "build_swarm_missions",
    "collect_capture_options_and_missions",
    "emit_evacuation_moves",
    "emit_followup_moves",
    "emit_rear_guard_moves",
    "enforce_inventory_cap",
    "process_multi_source_mission",
    "process_single_source_mission",
]
