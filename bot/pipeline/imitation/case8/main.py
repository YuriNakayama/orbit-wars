from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from policy.agent import agent  # noqa: E402

__all__ = ["agent"]
