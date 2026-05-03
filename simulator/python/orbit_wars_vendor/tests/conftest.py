"""Make the verbatim upstream `test_orbit_wars.py` work without
`kaggle_environments` installed.

The vendored test file imports symbols from
`kaggle_environments.envs.orbit_wars.orbit_wars`, which only exists when the
upstream `kaggle-environments` package is installed. Here we register an
import alias so the same import path resolves to our local
`orbit_wars_vendor.orbit_wars` module — letting the test file stay a verbatim
copy of upstream.
"""

from __future__ import annotations

import sys

import orbit_wars_vendor.orbit_wars as _vendored

# Register the upstream module path as an alias for the vendored module.
# This must run before pytest collects the test file.
sys.modules.setdefault("kaggle_environments", type(sys)("kaggle_environments"))
sys.modules.setdefault(
    "kaggle_environments.envs", type(sys)("kaggle_environments.envs")
)
sys.modules.setdefault(
    "kaggle_environments.envs.orbit_wars",
    type(sys)("kaggle_environments.envs.orbit_wars"),
)
sys.modules["kaggle_environments.envs.orbit_wars.orbit_wars"] = _vendored
