"""Unit + integration tests for src/dataset/executor.py."""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from dataset.schema import AgentSpec, AgentTiming, MatchSpec
from dataset.selfplay import executor


def test_percentile_handles_empty_list() -> None:
    assert executor._percentile([], 0.5) == 0.0


def test_percentile_returns_median_for_odd_sample() -> None:
    assert executor._percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)


def test_summarize_timings_counts_timeouts() -> None:
    timing = executor._summarize_timings([0.5, 0.8, 1.2, 1.5])
    assert timing.timeouts == 2
    assert timing.max == 1.5


def test_summarize_timings_empty_returns_zero() -> None:
    timing = executor._summarize_timings([])
    assert timing == AgentTiming(timeouts=0, p50=0.0, p95=0.0, max=0.0)


def test_episode_winner_single_active_player() -> None:
    state = [
        {"status": "ACTIVE", "reward": 0.0},
        {"status": "DONE", "reward": 0.0},
    ]
    assert executor._episode_winner(state) == 0


def test_episode_winner_draw_on_equal_rewards() -> None:
    state = [
        {"status": "DONE", "reward": 1.0},
        {"status": "DONE", "reward": 1.0},
    ]
    assert executor._episode_winner(state) == -1


def test_episode_winner_picks_highest_reward() -> None:
    state = [
        {"status": "DONE", "reward": 0.3},
        {"status": "DONE", "reward": 0.9},
        {"status": "DONE", "reward": 0.5},
    ]
    assert executor._episode_winner(state) == 1


def test_episode_winner_empty_is_draw() -> None:
    assert executor._episode_winner([]) == -1


def test_player_scores_sums_planets_and_fleets() -> None:
    class FakeEnv:
        steps = [
            [
                {
                    "observation": {
                        "player": 0,
                        "planets": [[0, 0, 0.0, 0.0, 1.0, 10, 1.0]],
                        "fleets": [[0, 0, 0.0, 0.0, 0.0, 0, 3]],
                    },
                },
                {
                    "observation": {
                        "player": 1,
                        "planets": [[1, 1, 0.0, 0.0, 1.0, 5, 1.0]],
                        "fleets": [],
                    },
                },
            ]
        ]

    scores = executor._player_scores(FakeEnv(), num_agents=2)
    assert scores == [13, 5]


def test_wrap_with_timing_records_elapsed() -> None:
    timings: list[float] = []

    def agent(_: Any) -> list[Any]:
        return []

    wrapped = executor._wrap_with_timing(agent, timings)
    wrapped({})
    assert len(timings) == 1
    assert timings[0] >= 0.0


def test_wrap_with_timing_passes_string_through() -> None:
    result = executor._wrap_with_timing("random", [])
    assert result == "random"


@pytest.mark.integration
def test_run_one_match_1v1_random_no_replay() -> None:
    pytest.importorskip("kaggle_environments")
    spec = MatchSpec(
        match_id="test_1v1_no_replay",
        run_id="run_test",
        mode="1v1",
        seed=0,
        agents=(
            AgentSpec(name="random", version="", callable_path="random"),
            AgentSpec(name="random", version="", callable_path="random"),
        ),
        save_replay=False,
        data_root="/tmp",
    )
    record, replay_bytes = executor.run_one_match(spec)
    assert replay_bytes is None
    assert record["mode"] == "1v1"
    assert record["turns"] > 0
    assert record["winner"] in {-1, 0, 1}
    assert len(record["agent_scores"]) == 2


@pytest.mark.integration
def test_run_one_match_with_replay_roundtrips() -> None:
    pytest.importorskip("kaggle_environments")
    spec = MatchSpec(
        match_id="test_1v1_with_replay",
        run_id="run_test",
        mode="1v1",
        seed=1,
        agents=(
            AgentSpec(name="random", version="", callable_path="random"),
            AgentSpec(name="random", version="", callable_path="random"),
        ),
        save_replay=True,
        data_root="/tmp",
    )
    record, replay_bytes = executor.run_one_match(spec)
    assert isinstance(replay_bytes, bytes)
    decoded = json.loads(gzip.decompress(replay_bytes).decode("utf-8"))
    assert "steps" in decoded
    assert record["replay_path"].endswith("test_1v1_with_replay.json.gz")
