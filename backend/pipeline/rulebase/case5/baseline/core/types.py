"""Shared dataclass / namedtuple types for the case5 agent.

Adapted from the LB1224 notebook by Roman Tamrazov (Apache License 2.0).
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[Any] = field(default_factory=list)
