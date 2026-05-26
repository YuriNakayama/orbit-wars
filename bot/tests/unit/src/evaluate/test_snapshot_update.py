"""Unit tests for src.evaluation.snapshot_update."""

from __future__ import annotations

import json
from pathlib import Path

from evaluate.snapshot_update import write_snapshot


def test_write_snapshot_creates_files(tmp_path: Path) -> None:
    obs = {"step": 5, "planets": [[1, 0, 50.0, 50.0, 1.0, 10, 1]]}
    action = [[1, 0.5, 5]]
    obs_out = tmp_path / "nested" / "obs.json"
    action_out = tmp_path / "nested" / "action.json"

    write_snapshot(obs, action, obs_out, action_out)

    assert obs_out.exists()
    assert action_out.exists()
    assert json.loads(obs_out.read_text()) == obs
    assert json.loads(action_out.read_text()) == action


def test_write_snapshot_overwrites_existing(tmp_path: Path) -> None:
    obs_out = tmp_path / "obs.json"
    action_out = tmp_path / "action.json"
    obs_out.write_text("old")
    action_out.write_text("old")

    obs = {"step": 1}
    action = [[2, 0.0, 1]]
    write_snapshot(obs, action, obs_out, action_out)

    assert json.loads(obs_out.read_text()) == obs
    assert json.loads(action_out.read_text()) == action


def test_write_snapshot_empty_action(tmp_path: Path) -> None:
    obs_out = tmp_path / "obs.json"
    action_out = tmp_path / "action.json"
    write_snapshot({"step": 0}, [], obs_out, action_out)
    assert json.loads(action_out.read_text()) == []
