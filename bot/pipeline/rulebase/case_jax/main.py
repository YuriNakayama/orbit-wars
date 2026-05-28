# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Local entry point for case_jax (Python oracle agent).

case_jax is a development case for the fully-JAX rulebase agent; the `baseline/`
body is the case8 Python agent kept as the parity oracle. It exposes `agent` so
the standard self-play / evaluation harness can drive the oracle directly. The
JAX agent lives in `baseline_jax/` and is consumed by the JAX rollout, not by
this entry point. Kaggle submission remains case8's responsibility (see README).

Adding the current working directory to `sys.path` makes `baseline/` importable
as a top-level package; the baseline internals use relative imports so they
resolve under both `pipeline.rulebase.case_jax.baseline` (local) and `baseline`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
