# Adapted from "orbit-star-wars-lb-max-1224" by Roman Tamrazov
# https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224
# Licensed under Apache License 2.0
"""Kaggle submission entry point for case5.

Kaggle executes this file directly; the `pipeline.rulebase.case5` parent
package does not exist in the submission runtime. Adding the current working
directory to `sys.path` makes `baseline/` importable as a top-level package,
while the baseline internals use relative imports so they resolve under both
`pipeline.rulebase.case5.baseline` (local) and `baseline` (Kaggle).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
