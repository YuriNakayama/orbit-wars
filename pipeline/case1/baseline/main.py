# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Kaggle submission entry point — exposes agent(obs) at module top level."""

from __future__ import annotations

from pipeline.case1.baseline.agent import agent

__all__ = ["agent"]
