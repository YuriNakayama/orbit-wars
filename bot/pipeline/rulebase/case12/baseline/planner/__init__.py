"""Naïve MCTS planner for case12.

Combinatorial Multi-armed Bandit (CMAB) sampling per Ontañón 2017.
"""

from __future__ import annotations

from .naive_mcts import run_naive_mcts

__all__ = ["run_naive_mcts"]
