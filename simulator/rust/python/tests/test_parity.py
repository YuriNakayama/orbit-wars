"""End-to-end parity between the Python (vendored) and Rust interpreters.

This test runs **two independent envs** with the same seed and the same
deterministic action sequence, and verifies that both observe the same
state at every turn. Crucially we do NOT inject one backend's state into
the other — each backend is responsible for its own initial generation
and comet spawning. The Rust facade delegates generation-touching turns
back to Python (see `_facade.py`), so a passing test here guarantees:

  1. Both backends start from the same planets/comets given the same seed
     (because both call into the same `random` global state via the upstream
     Python `generate_planets` / `generate_comet_paths`).
  2. The Rust per-step physics agrees with Python bit-exactly on
     non-spawn turns.
  3. Comet spawn events fire on both sides at exactly the same turns.

`relative tolerance 1e-9` absorbs cross-platform fmadd ordering noise but
keeps the regression signal sharp.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

# Force-register the orbit_wars env name with the facade BEFORE the test
# imports anything from kaggle_environments.
import orbit_wars_rust

from kaggle_environments import make


REL_TOL = 1e-9
ABS_TOL = 1e-9


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _planet_key(p: list[Any]) -> tuple[int, int]:
    return (int(p[0]), int(p[1]))


def _fleet_key(f: list[Any]) -> tuple[int, int]:
    return (int(f[0]), int(f[1]))


def _assert_planets_equal(a: list[list[Any]], b: list[list[Any]]) -> None:
    assert len(a) == len(b), f"planet count differs: {len(a)} vs {len(b)}"
    a_sorted = sorted(a, key=_planet_key)
    b_sorted = sorted(b, key=_planet_key)
    for pa, pb in zip(a_sorted, b_sorted, strict=True):
        assert int(pa[0]) == int(pb[0])
        assert int(pa[1]) == int(pb[1])
        for i in (2, 3, 4):  # x, y, radius
            assert _close(float(pa[i]), float(pb[i])), (
                f"planet[{int(pa[0])}] field {i} mismatch: {pa} vs {pb}"
            )
        assert int(pa[5]) == int(pb[5]), f"planet[{int(pa[0])}].ships diverged"
        assert int(pa[6]) == int(pb[6])


def _assert_fleets_equal(a: list[list[Any]], b: list[list[Any]]) -> None:
    assert len(a) == len(b), f"fleet count differs: {len(a)} vs {len(b)}"
    a_sorted = sorted(a, key=_fleet_key)
    b_sorted = sorted(b, key=_fleet_key)
    for fa, fb in zip(a_sorted, b_sorted, strict=True):
        assert int(fa[0]) == int(fb[0])
        assert int(fa[1]) == int(fb[1])
        for i in (2, 3, 4):  # x, y, angle
            assert _close(float(fa[i]), float(fb[i])), f"fleet field {i} mismatch: {fa} vs {fb}"
        assert int(fa[5]) == int(fb[5])
        assert int(fa[6]) == int(fb[6])


def _build_env(backend: str, seed: int, num_agents: int) -> Any:
    """Construct an env with the given backend.

    The upstream `generate_planets` / `generate_comet_paths` consume Python's
    *global* `random` state, so we have to reset that state immediately
    before each `make`/`reset`/`step` that might trigger generation. The
    parity harness in `_run_two_envs` snapshots and restores the global
    state across the two `env.step()` calls so both backends draw from the
    same RNG cursor at every turn.
    """
    orbit_wars_rust.set_backend(backend)
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=num_agents)
    return env


class _SyncedRandom:
    """Context manager that snapshots Python's global random state before
    `env.step()` and restores it for the second backend, so both consume the
    same RNG draws during initial generation / comet spawn."""

    def __init__(self) -> None:
        self._snapshot: tuple[Any, ...] | None = None

    def snapshot(self) -> None:
        self._snapshot = random.getstate()

    def restore(self) -> None:
        assert self._snapshot is not None, "snapshot before restore"
        random.setstate(self._snapshot)


def _step_pair(
    env_py: Any,
    env_rs: Any,
    actions: list[list[list[Any]]],
    rng_sync: _SyncedRandom,
) -> None:
    """Step both envs with the same RNG cursor.

    Python global random is mutated by `generate_planets` (turn 0) and
    `generate_comet_paths` (turns 49/149/...). The Rust facade delegates
    those turns back to Python, so both backends pull from `random.*`. To
    keep them aligned we snapshot before the Python env steps and restore
    before the Rust env steps.
    """
    rng_sync.snapshot()
    env_py.step(actions)
    rng_sync.restore()
    env_rs.step(actions)


def _seeded_envs(seed: int, num_agents: int) -> tuple[Any, Any, _SyncedRandom]:
    """Build matched envs: same RNG seed, same first init draw."""
    random.seed(seed)
    env_py = _build_env("python", seed, num_agents)
    rng_sync = _SyncedRandom()
    # Python env consumed entropy during reset; reseed and create the rust
    # env so it sees a fresh stream — but BOTH `step(0)` (which generates
    # planets) must see the same `random` state. Reseed once more here.
    random.seed(seed)
    env_rs = _build_env("rust", seed, num_agents)
    random.seed(seed)
    return env_py, env_rs, rng_sync


def _scripted_actions(seed: int, num_agents: int, turn: int, planets: list[list[Any]]) -> list[list[list[Any]]]:
    """Reproducible per-player actions independent of which backend we ran."""
    rng = random.Random(seed * 100003 + turn)
    actions: list[list[list[Any]]] = [[] for _ in range(num_agents)]
    for p in planets:
        owner = int(p[1])
        ships = int(p[5])
        if 0 <= owner < num_agents and ships >= 20 and rng.random() < 0.3:
            actions[owner].append([int(p[0]), rng.uniform(0, 2 * math.pi), ships // 2])
    return actions


def _compare_step(
    env_py: Any,
    env_rs: Any,
    turn: int,
    seed: int,
) -> None:
    py_obs = env_py.steps[-1][0]["observation"]
    rs_obs = env_rs.steps[-1][0]["observation"]
    py_planets = py_obs["planets"] if isinstance(py_obs, dict) else py_obs.planets
    rs_planets = rs_obs["planets"] if isinstance(rs_obs, dict) else rs_obs.planets
    py_fleets = py_obs["fleets"] if isinstance(py_obs, dict) else py_obs.fleets
    rs_fleets = rs_obs["fleets"] if isinstance(rs_obs, dict) else rs_obs.fleets
    try:
        _assert_planets_equal(py_planets, rs_planets)
        _assert_fleets_equal(py_fleets, rs_fleets)
    except AssertionError as e:
        raise AssertionError(f"divergence at turn={turn}, seed={seed}: {e}") from e


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_parity_noop_short(seed: int) -> None:
    """30-turn parity over noop actions — fast smoke test (covers comet
    spawn at step 50 if it falls inside the window)."""
    env_py, env_rs, rng_sync = _seeded_envs(seed, num_agents=2)

    for turn in range(1, 31):
        _step_pair(env_py, env_rs, [[], []], rng_sync)
        if env_py.done or env_rs.done:
            break
        _compare_step(env_py, env_rs, turn, seed)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_parity_random_actions_with_first_comet(seed: int) -> None:
    """60-turn parity with deterministic random actions. Covers the first
    comet spawn at step 50, which the previous test missed because the
    Rust interpreter never fired it."""
    env_py, env_rs, rng_sync = _seeded_envs(seed, num_agents=2)

    for turn in range(1, 61):
        py_obs = env_py.steps[-1][0]["observation"]
        py_planets = py_obs["planets"] if isinstance(py_obs, dict) else py_obs.planets
        actions = _scripted_actions(seed, 2, turn, py_planets)

        _step_pair(env_py, env_rs, actions, rng_sync)
        if env_py.done or env_rs.done:
            break
        _compare_step(env_py, env_rs, turn, seed)


def test_parity_comet_spawn_events() -> None:
    """Direct check that the Rust backend fires the same comet spawn events
    as Python. Targets the previously-undetected gap where the Rust
    interpreter had no comet generation path at all."""
    env_py, env_rs, rng_sync = _seeded_envs(seed=0, num_agents=2)

    events: dict[str, list[tuple[int, list[int]]]] = {"python": [], "rust": []}
    prev_cids: dict[str, set[int]] = {"python": set(), "rust": set()}
    for _ in range(500):
        _step_pair(env_py, env_rs, [[], []], rng_sync)
        if env_py.done or env_rs.done:
            break
        for label, env in (("python", env_py), ("rust", env_rs)):
            obs = env.steps[-1][0]["observation"]
            cids = set(obs.get("comet_planet_ids") or [])
            new = cids - prev_cids[label]
            if new:
                events[label].append((int(obs.get("step")), sorted(new)))
            prev_cids[label] = cids

    expected_steps = [50, 150, 250, 350, 450]
    py = events["python"]
    rs = events["rust"]
    assert [s for s, _ in py] == expected_steps, f"upstream broke spawn cadence: {py}"
    assert py == rs, (
        f"Rust missed comet spawn events. python={py} rust={rs}. "
        "Likely regression in `_facade._is_comet_spawn_turn` or upstream."
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_full_episode(seed: int) -> None:
    """500-turn full-episode parity. Slow; nightly only."""
    env_py, env_rs, rng_sync = _seeded_envs(seed, num_agents=2)

    for turn in range(1, 501):
        py_obs = env_py.steps[-1][0]["observation"]
        py_planets = py_obs["planets"] if isinstance(py_obs, dict) else py_obs.planets
        actions = _scripted_actions(seed, 2, turn, py_planets)

        _step_pair(env_py, env_rs, actions, rng_sync)
        if env_py.done or env_rs.done:
            break
        _compare_step(env_py, env_rs, turn, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_parity_4p_full_episode(seed: int) -> None:
    """4-player FFA full episode parity."""
    env_py, env_rs, rng_sync = _seeded_envs(seed, num_agents=4)

    for turn in range(1, 501):
        py_obs = env_py.steps[-1][0]["observation"]
        py_planets = py_obs["planets"] if isinstance(py_obs, dict) else py_obs.planets
        actions = _scripted_actions(seed, 4, turn, py_planets)

        _step_pair(env_py, env_rs, actions, rng_sync)
        if env_py.done or env_rs.done:
            break
        _compare_step(env_py, env_rs, turn, seed)
