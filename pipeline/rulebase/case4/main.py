# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Kaggle submission entry point for case4.

Kaggle executes this file directly; the `pipeline.rulebase.case4` parent
package does not exist in the submission runtime. Adding the current working
directory to `sys.path` makes `baseline/` importable as a top-level package,
while the baseline internals use relative imports so they resolve under both
`pipeline.rulebase.case4.baseline` (local) and `baseline` (Kaggle).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
