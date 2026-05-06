"""Stub agent for imitation/case0 — RunPod E2E smoke only.

Returns NO_OP every turn. The case0 pipeline is not intended for play strength,
only for verifying the RunPod training basis and submit packager structure.
"""

from __future__ import annotations

from typing import Any


def agent(obs: Any) -> list[list[Any]]:
    """No-op policy: send no fleets."""
    del obs
    return []


__all__ = ["agent"]
