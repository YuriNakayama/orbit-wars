"""Parity test for featurizer refactor.

The reinforce/case1 featurizer is called with bit-exact requirements because
the BC warm-start (`case9_per_planet/best.pt`) was trained on its current
output. Any refactor that drifts in float32 LSBs breaks the warm-start.

This test runs a deterministic 5-turn rollout with the noop opponent, captures
each turn's obs, runs `featurize` on the current implementation, and asserts
that all output tensors are bit-equal across multiple invocations. The
fixture cache (under `_PARITY_CACHE_PATH`) is generated on first run and
checked into git so subsequent runs assert against the recorded baseline.

Regeneration: delete the cache file and re-run the test.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from orbit_wars_sim import make_orbit_wars_env

from pipeline.reinforce.case1.policy.featurizer import (
    HistoryState,
    featurize,
    update_history,
)
from pipeline.reinforce.case1.policy.types import BatchFeatures

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_PARITY_CACHE_PATH = _FIXTURE_DIR / "featurize_parity_baseline.pt"
_OBS_CACHE_PATH = _FIXTURE_DIR / "featurize_parity_obs.pkl"

PARITY_SEED = 0
PARITY_TURNS = 60


def _fire_from_home_each_turn(
    obs: dict[str, Any], seat: int
) -> list[list[int | float]]:
    """Tiny rule: own first planet fires 1 ship at a 0.5-rad heading every turn.

    Used only to populate the fixture with non-empty fleet lists so the
    fleet×planet vectorization path is exercised by the parity test.
    """
    for row in obs.get("planets", []) or []:
        _pid, owner, _x, _y, _r, ships, _prod = row
        if int(owner) == seat and int(ships) >= 2:
            return [[int(_pid), 0.5, 1]]
    return []


def _collect_obs(seed: int, turns: int) -> list[dict[str, Any]]:
    """Run a deterministic rollout with periodic launches.

    Both seats fire 1 ship from their home planet every turn (when they have
    ≥2 ships), producing a realistic mix of in-flight fleets that exercise
    the featurizer's fleet×planet inner loop.
    """
    env = make_orbit_wars_env(agents=2, seed=seed)
    obs_log: list[dict[str, Any]] = []
    for _ in range(turns):
        if env.done:
            break
        seat0_obs = dict(env.steps[-1][0]["observation"])
        seat1_obs = dict(env.steps[-1][1]["observation"])
        obs_log.append(seat0_obs)
        a0 = _fire_from_home_each_turn(seat0_obs, seat=0)
        a1 = _fire_from_home_each_turn(seat1_obs, seat=1)
        env.step([a0, a1])
    return obs_log


def _featurize_all(obs_log: list[dict[str, Any]]) -> list[BatchFeatures]:
    history = HistoryState()
    out: list[BatchFeatures] = []
    for obs in obs_log:
        batch, _snap = featurize(obs, history)
        out.append(batch)
        update_history(history, obs, actions=None)
    return out


def _batch_to_dict(batch: BatchFeatures) -> dict[str, torch.Tensor]:
    return {
        "planet_feats": batch.planet_feats,
        "planet_mask": batch.planet_mask,
        "my_planet_mask": batch.my_planet_mask,
        "target_mask": batch.target_mask,
        "global_feats": batch.global_feats,
        "template_ctx": batch.template_ctx,
        "candidate_feats": batch.candidate_feats,
        "candidate_mask": batch.candidate_mask,
        "candidate_pid": batch.candidate_pid,
    }


def _ensure_fixtures() -> None:
    """Generate the obs and baseline featurize cache if missing."""
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if not _OBS_CACHE_PATH.exists():
        obs_log = _collect_obs(PARITY_SEED, PARITY_TURNS)
        with _OBS_CACHE_PATH.open("wb") as f:
            pickle.dump(obs_log, f)
    if not _PARITY_CACHE_PATH.exists():
        with _OBS_CACHE_PATH.open("rb") as f:
            obs_log = pickle.load(f)
        baseline = [_batch_to_dict(b) for b in _featurize_all(obs_log)]
        torch.save(baseline, _PARITY_CACHE_PATH)


@pytest.fixture(scope="module")
def parity_baseline() -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
    _ensure_fixtures()
    with _OBS_CACHE_PATH.open("rb") as f:
        obs_log = pickle.load(f)
    baseline = torch.load(_PARITY_CACHE_PATH, weights_only=False)
    return obs_log, baseline


def test_featurize_matches_baseline(
    parity_baseline: tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]],
) -> None:
    obs_log, baseline = parity_baseline
    current = [_batch_to_dict(b) for b in _featurize_all(obs_log)]

    assert len(current) == len(baseline), (
        f"turn count drift: current={len(current)} baseline={len(baseline)}"
    )

    for turn, (cur, base) in enumerate(zip(current, baseline, strict=True)):
        assert cur.keys() == base.keys(), f"turn {turn}: key drift"
        for name in cur:
            c = cur[name]
            b = base[name]
            assert c.shape == b.shape, (
                f"turn {turn} field {name}: shape {c.shape} vs {b.shape}"
            )
            assert c.dtype == b.dtype, (
                f"turn {turn} field {name}: dtype {c.dtype} vs {b.dtype}"
            )
            if c.dtype.is_floating_point:
                if not torch.equal(c, b):
                    max_abs = (c - b).abs().max().item()
                    raise AssertionError(
                        f"turn {turn} field {name}: float drift, "
                        f"max_abs_diff={max_abs:.3e}"
                    )
            else:
                if not torch.equal(c, b):
                    raise AssertionError(
                        f"turn {turn} field {name}: int/bool tensors differ"
                    )
