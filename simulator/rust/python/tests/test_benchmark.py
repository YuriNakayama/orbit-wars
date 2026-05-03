"""Wall-clock benchmark: Rust vs Python interpreter on the same env shape.

Not a CI gate — runner specs vary too much. Engineers run locally:

    cd bot
    uv run pytest ../simulator/rust/python/tests/test_benchmark.py -s -m slow

Three benches live here, all sized to 30 matches × 2 agents to keep
variance under ±15% on shared hosts:

  * `test_benchmark_speedup_over_python` — `env.run` legacy path on
    python vs rust backend. Tracks the transparent (caller-unchanged)
    speedup ceiling, currently ~2× on M-series Mac.

  * `test_benchmark_run_episode_vs_env_run` — the new opt-in
    `orbit_wars_rust.run_episode(env, agents)` API vs the same env's
    `env.run(agents)`. Caller swap is one import + one call.

  * `test_benchmark_run_episodes_helper` — `run_episodes(env, agents,
    seeds)` (env reuse) vs make+run_episode per seed. Quantifies the
    cost saved by reusing the kaggle_environments compiled schema.

`docs/plans/rust-simulator/benchmark_results.md` is updated by hand
whenever the numbers shift materially.
"""

from __future__ import annotations

import time

import pytest

import orbit_wars_rust
from kaggle_environments import make


def _run_via_env_run(backend: str, num_episodes: int, num_agents: int) -> float:
    """Time `env.run` for `num_episodes` games on the given backend."""
    orbit_wars_rust.set_backend(backend)
    t0 = time.perf_counter()
    for seed in range(num_episodes):
        env = make("orbit_wars", configuration={"agents": num_agents, "seed": seed})
        env.run(["random"] * num_agents)
    return time.perf_counter() - t0


def _run_via_run_episode(num_episodes: int, num_agents: int) -> float:
    """Time `orbit_wars_rust.run_episode` (rust backend, opt-in)."""
    orbit_wars_rust.set_backend("rust")
    t0 = time.perf_counter()
    for seed in range(num_episodes):
        env = make("orbit_wars", configuration={"agents": num_agents, "seed": seed})
        orbit_wars_rust.run_episode(env, ["random"] * num_agents)
    return time.perf_counter() - t0


@pytest.mark.slow
def test_benchmark_speedup_over_python() -> None:
    """Transparent speedup: same `env.run` call, python vs rust backend.

    This is the speedup callers get **without changing any code** other
    than `import orbit_wars_rust; orbit_wars_rust.use_rust()`. Floor
    >=1.5× catches the case where the extension module didn't load.
    """
    num_episodes = 30
    num_agents = 2

    py_time = _run_via_env_run("python", num_episodes, num_agents)
    rs_time = _run_via_env_run("rust", num_episodes, num_agents)
    speedup = py_time / max(rs_time, 1e-9)

    print(
        f"\n[env.run] {num_episodes} episodes / {num_agents} agents:"
        f" python={py_time:.3f}s, rust={rs_time:.3f}s, speedup={speedup:.2f}x"
    )
    assert speedup >= 1.5, (
        f"speedup floor failed: rust={rs_time:.3f}s python={py_time:.3f}s ratio={speedup:.2f}x"
    )


@pytest.mark.slow
def test_benchmark_run_episode_vs_env_run() -> None:
    """Opt-in `run_episode` API vs `env.run` (both on rust backend).

    Quantifies what the caller gains by replacing one function call
    (`env.run` → `orbit_wars_rust.run_episode`). The gap is the
    kaggle_environments per-step framework overhead (`structify`,
    `deepcopy`, `process_schema`) that `run_episode` bypasses.
    """
    num_episodes = 30
    num_agents = 2

    legacy = _run_via_env_run("rust", num_episodes, num_agents)
    batch = _run_via_run_episode(num_episodes, num_agents)
    speedup = legacy / max(batch, 1e-9)

    print(
        f"\n[run_episode] {num_episodes} episodes / {num_agents} agents:"
        f" env.run={legacy:.3f}s, run_episode={batch:.3f}s, speedup={speedup:.2f}x"
    )
    # Floor at 5× — local M-series typically sees 12-15×; below 5×
    # suggests the batch path silently fell back to env.run.
    assert speedup >= 5.0, (
        f"run_episode speedup floor failed: env.run={legacy:.3f}s "
        f"run_episode={batch:.3f}s ratio={speedup:.2f}x"
    )


@pytest.mark.slow
def test_benchmark_run_episodes_parallel() -> None:
    """Tier 6: caller hands off seeds to a multi-process Pool helper.

    `run_episodes_parallel` removes the per-process backend toggle and
    Pool plumbing from caller code. Compares serial (parallel=1) vs
    fork-pooled (parallel=4/8) on the same Python interpreter — the
    speedup is the multi-process fanout, not extra single-process gain.
    """
    num_episodes = 30
    num_agents = 2

    serial = time.perf_counter()
    orbit_wars_rust.run_episodes_parallel(
        agents=["random"] * num_agents,
        seeds=list(range(num_episodes)),
        parallel=1,
    )
    serial_t = time.perf_counter() - serial

    pool_t = time.perf_counter()
    orbit_wars_rust.run_episodes_parallel(
        agents=["random"] * num_agents,
        seeds=list(range(num_episodes)),
        parallel=8,
        mp_context="fork",
    )
    pool_t = time.perf_counter() - pool_t

    speedup = serial_t / max(pool_t, 1e-9)
    print(
        f"\n[run_episodes_parallel] {num_episodes} matches / {num_agents} agents:"
        f" serial={serial_t:.3f}s, fork×8={pool_t:.3f}s, speedup={speedup:.2f}x"
    )
    # Floor at 3× — even noisy hosts with few cores see >=3× from
    # batch+fork. M-series Mac typically hits 5-7× here.
    assert speedup >= 3.0, (
        f"parallel speedup floor failed: serial={serial_t:.3f}s "
        f"fork×8={pool_t:.3f}s ratio={speedup:.2f}x"
    )


@pytest.mark.slow
def test_benchmark_run_episodes_helper() -> None:
    """Quantify the env-reuse helper `run_episodes(env, agents, seeds)`.

    `make()` rebuilds the JSON schema each call (~40 ms). Reusing the
    same env across seeds keeps it cached; only the per-reset jsonschema
    validate (~10 ms / match) remains.
    """
    num_episodes = 30
    num_agents = 2
    orbit_wars_rust.set_backend("rust")

    # Warmup — exclude first-import cost from both sides.
    warm = make("orbit_wars", configuration={"agents": num_agents, "seed": 99999})
    orbit_wars_rust.run_episode(warm, ["random"] * num_agents)

    # A) make + run_episode each seed.
    t0 = time.perf_counter()
    for seed in range(num_episodes):
        env = make("orbit_wars", configuration={"agents": num_agents, "seed": seed})
        orbit_wars_rust.run_episode(env, ["random"] * num_agents)
    make_each = time.perf_counter() - t0

    # B) run_episodes helper (env reuse).
    env_reuse = make("orbit_wars", configuration={"agents": num_agents, "seed": 0})
    t0 = time.perf_counter()
    orbit_wars_rust.run_episodes(env_reuse, ["random"] * num_agents, list(range(num_episodes)))
    reuse = time.perf_counter() - t0

    speedup = make_each / max(reuse, 1e-9)
    print(
        f"\n[run_episodes] {num_episodes} matches / {num_agents} agents:"
        f" make_each={make_each:.3f}s, env_reuse={reuse:.3f}s, speedup={speedup:.2f}x"
    )
    # Floor at 0.9× — make() per-call cost is small (~40 ms on this
    # schema) and env-reuse only saves that. On noisy hosts the gap is
    # routinely lost in measurement variance, so we only gate against
    # outright regression rather than a positive speedup.
    assert speedup >= 0.9, (
        f"env-reuse regressed: make_each={make_each:.3f}s reuse={reuse:.3f}s ratio={speedup:.2f}x"
    )
