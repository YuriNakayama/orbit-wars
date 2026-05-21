"""Self-play executor integration tests."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable

from dataset.schema import MatchSpec
from dataset.selfplay import executor


def test_run_one_match_1v1_random_no_replay(
    random_match_spec_factory: Callable[..., MatchSpec],
) -> None:
    spec = random_match_spec_factory(match_id="test_1v1_no_replay")
    record, replay_bytes = executor.run_one_match(spec)
    assert replay_bytes is None
    assert record["mode"] == "1v1"
    assert record["turns"] > 0
    assert record["winner"] in {-1, 0, 1}
    assert len(record["agent_scores"]) == 2


def test_run_one_match_with_replay_roundtrips(
    random_match_spec_factory: Callable[..., MatchSpec],
) -> None:
    spec = random_match_spec_factory(
        match_id="test_1v1_with_replay",
        seed=1,
        save_replay=True,
    )
    record, replay_bytes = executor.run_one_match(spec)
    assert isinstance(replay_bytes, bytes)
    decoded = json.loads(gzip.decompress(replay_bytes).decode("utf-8"))
    assert "steps" in decoded
    assert record["replay_uri"].endswith("test_1v1_with_replay.json.gz")
