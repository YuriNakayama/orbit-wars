"""Smoke tests for imitation/case8 featurizer + candidate block."""

from __future__ import annotations

import pytest

from pipeline.imitation.case8.policy.candidates import (
    CAND_FEAT_DIM,
    CAND_K,
    build_candidate_block,
)
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
    update_history,
)


def _planet_row(
    pid: int,
    owner: int,
    x: float,
    y: float,
    radius: float = 1.5,
    ships: int = 10,
    production: int = 1,
) -> list[float]:
    return [pid, owner, x, y, radius, ships, production]


def _make_obs() -> dict[str, object]:
    planets = [
        _planet_row(0, 0, 20.0, 50.0, ships=10, production=2),  # mine
        _planet_row(1, 1, 80.0, 50.0, ships=15, production=2),  # enemy
        _planet_row(2, -1, 50.0, 20.0, ships=5, production=1),  # neutral
        _planet_row(3, -1, 50.0, 80.0, ships=8, production=1),  # neutral
        _planet_row(4, 0, 30.0, 30.0, ships=3, production=1),  # mine
    ]
    return {
        "step": 5,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "angular_velocity": 0.05,
    }


def test_featurize_shape() -> None:
    obs = _make_obs()
    history = HistoryState()
    batch, snap = featurize(obs, history)
    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)
    assert batch.candidate_feats.shape == (1, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    assert batch.candidate_mask.shape == (1, MAX_PLANETS, CAND_K)
    assert batch.candidate_pid.shape == (1, MAX_PLANETS, CAND_K)
    assert snap.player == 0
    assert 0 in snap.my_planet_ids
    assert 4 in snap.my_planet_ids


def test_no_op_slot_always_valid() -> None:
    obs = _make_obs()
    batch, snap = featurize(obs, HistoryState())
    # slot 0 of every my-planet should have candidate_mask True
    for slot, pid in enumerate(snap.planet_ids):
        if pid in snap.my_planet_ids:
            assert bool(batch.candidate_mask[0, slot, 0].item()) is True
            # is_valid feature for slot 0 == 1.0
            assert batch.candidate_feats[0, slot, 0, 0].item() == pytest.approx(1.0)


def test_candidate_block_shapes() -> None:
    src_row = _planet_row(0, 0, 20.0, 50.0, ships=30)
    planets = [
        src_row,
        _planet_row(1, 1, 80.0, 50.0, ships=15),
        _planet_row(2, -1, 50.0, 20.0, ships=5),
    ]
    feats, mask, pids = build_candidate_block(src_row, planets, player=0)
    assert feats.shape == (CAND_K, CAND_FEAT_DIM)
    assert mask.shape == (CAND_K,)
    assert len(pids) == CAND_K
    # slot 0 = no-op, valid by definition
    assert bool(mask[0]) is True
    assert pids[0] == -1


def test_update_history_records_launches() -> None:
    history = HistoryState()
    obs = _make_obs()
    update_history(history, obs, actions=[[0, 1.5, 5]])
    assert len(history.recent_launches) == 1
    assert history.recent_launches[0].owner == 0
    assert history.recent_launches[0].ships == 5
