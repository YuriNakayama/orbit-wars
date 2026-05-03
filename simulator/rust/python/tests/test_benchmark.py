"""Wall-clock benchmark: Rust vs Python interpreter on the same env shape.

We do NOT make this a CI gate — runner specs vary too much. Instead we leave
a printable comparison that engineers can run locally:

    cd bot
    uv run pytest ../simulator/rust/python/tests/test_benchmark.py -s -m slow

The 10x target lives in `02-requirements.md`; the actual numbers and the
machine they were measured on get pasted into
`docs/plans/rust-simulator/benchmark_results.md`.
"""

from __future__ import annotations

import os
import time

import pytest

import orbit_wars_rust  # noqa: F401  registers facade
from kaggle_environments import make


def _run_episodes(backend: str, num_episodes: int, num_agents: int) -> float:
    os.environ["ORBIT_WARS_BACKEND"] = backend
    t0 = time.perf_counter()
    for seed in range(num_episodes):
        env = make("orbit_wars", configuration={"agents": num_agents, "seed": seed})
        env.run(["random"] * num_agents)
    return time.perf_counter() - t0


@pytest.mark.slow
def test_benchmark_speedup_over_python() -> None:
    """Print Python vs Rust wall-clock and assert >=2x as a sanity floor."""
    num_episodes = 5
    num_agents = 2

    py_time = _run_episodes("python", num_episodes, num_agents)
    rs_time = _run_episodes("rust", num_episodes, num_agents)
    speedup = py_time / max(rs_time, 1e-9)

    print(
        f"\n[benchmark] {num_episodes} episodes / {num_agents} agents:"
        f" python={py_time:.3f}s, rust={rs_time:.3f}s, speedup={speedup:.2f}x"
    )

    # We aim for 10x in `02-requirements.md`. Floor at 2x to keep the CI
    # signal alive even on busy hosts; below this we know something is
    # genuinely wrong (extension-module not loaded, pyo3 fallback, etc.).
    assert speedup >= 2.0, (
        f"speedup floor failed: rust={rs_time:.3f}s python={py_time:.3f}s ratio={speedup:.2f}x"
    )
