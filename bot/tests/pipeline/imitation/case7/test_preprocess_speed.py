"""case7 preprocess: vectorization + multiprocessing の sanity check。

featurize の vectorized path と multiprocessing pool が、frame 出力を変えずに
動くことを smoke 確認する。性能数値は assert しない (環境依存)。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from pipeline.imitation.case7.training.preprocess import (
    _episode_worker,
    _iter_episode_frames,
)


def _write_synthetic_replay(path: Path, n_steps: int = 8) -> None:
    """2-player 1v1 replay の最小構造。fleet が turn ごとに増えていく形。"""
    steps = []
    for step_idx in range(n_steps):
        # 各 player の obs にひとつずつ planet + 数件の fleet を入れる
        planets = [
            [0, 0, 30.0, 50.0, 1.5, 10 + step_idx, 1],
            [1, 1, 70.0, 50.0, 1.5, 10 + step_idx, 1],
        ]
        fleets = [
            [step_idx * 10 + j, j % 2, 40.0 + j, 50.0, 0.0, j, 5]
            for j in range(min(step_idx, 5))
        ]
        step_payload = []
        for slot in range(2):
            obs = {
                "step": step_idx,
                "player": slot,
                "planets": planets,
                "fleets": fleets,
                "comet_planet_ids": [],
                "angular_velocity": 0.05,
            }
            step_payload.append(
                {
                    "observation": obs,
                    "action": [[0, 0.5, 1]] if slot == 0 and step_idx > 0 else [],
                }
            )
        steps.append(step_payload)
    with gzip.open(path, "wt") as f:
        json.dump({"steps": steps}, f)


def test_iter_episode_frames_smoke(tmp_path: Path) -> None:
    """生 featurize 経路で 1 episode を走らせ、frames が返ることを確認。"""
    replay = tmp_path / "replay.json.gz"
    _write_synthetic_replay(replay, n_steps=6)
    frames = _iter_episode_frames(replay, [0, 1])
    assert len(frames) > 0
    assert all("planet_feats" in f for f in frames)


def test_episode_worker_returns_match_id_and_frames(tmp_path: Path) -> None:
    """multiprocessing 用 worker が (match_id, frames) を返すことを確認。"""
    replay = tmp_path / "replay.json.gz"
    _write_synthetic_replay(replay, n_steps=4)
    match_id, frames = _episode_worker((str(replay), [0, 1], "match-test"))
    assert match_id == "match-test"
    assert len(frames) > 0


def test_episode_worker_handles_missing_replay(tmp_path: Path) -> None:
    """存在しない replay path でも例外を出さず空 frames を返す。"""
    missing = tmp_path / "nope.json.gz"
    match_id, frames = _episode_worker((str(missing), [0, 1], "missing"))
    assert match_id == "missing"
    assert frames == []


@pytest.mark.parametrize("n_steps", [3, 12])
def test_vectorized_featurize_consistent_across_episode_lengths(
    tmp_path: Path, n_steps: int
) -> None:
    """異なる長さの episode で featurize が一貫した shape を返す。"""
    replay = tmp_path / f"replay_{n_steps}.json.gz"
    _write_synthetic_replay(replay, n_steps=n_steps)
    frames = _iter_episode_frames(replay, [0, 1])
    assert len(frames) > 0
    # 全 frame で shape が一致
    expected_planet_dim = len(frames[0]["planet_feats"])
    expected_global_dim = len(frames[0]["global_feats"])
    for f in frames:
        assert len(f["planet_feats"]) == expected_planet_dim
        assert len(f["global_feats"]) == expected_global_dim
