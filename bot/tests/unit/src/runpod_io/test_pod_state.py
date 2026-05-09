"""runpod_io.pod_state の正規化と integral helper の単体テスト。"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from runpod_io import pod_state


def test_to_state_normalizes_payload(sample_pod_payload: dict[str, Any]) -> None:
    state = pod_state._to_state(sample_pod_payload)
    assert state.pod_id == "pod-abc"
    assert state.desired_status == "RUNNING"
    assert state.is_active is True
    assert state.cost_per_hr == pytest.approx(0.45)
    assert state.uptime_seconds == 3725
    assert state.gpu_display_name == "RTX 3090"
    assert state.estimated_cost_usd == pytest.approx(0.45 * 3725 / 3600)


def test_to_state_handles_missing_fields() -> None:
    state = pod_state._to_state({"id": "pod-x"})
    assert state.pod_id == "pod-x"
    assert state.desired_status == "UNKNOWN"
    assert state.cost_per_hr == 0.0
    assert state.uptime_seconds == 0
    assert state.gpu_display_name == ""
    assert state.is_active is False


def test_list_pods_filters_via_sdk(sample_pod_payload: dict[str, Any]) -> None:
    with patch(
        "runpod_io.pod_state.runpod_sdk.get_pods", return_value=[sample_pod_payload]
    ):
        states = pod_state.list_pods()
    assert len(states) == 1
    assert states[0].pod_id == "pod-abc"


def test_list_pods_returns_empty_when_none() -> None:
    with patch("runpod_io.pod_state.runpod_sdk.get_pods", return_value=None):
        assert pod_state.list_pods() == []


def test_get_pod_returns_none_when_missing() -> None:
    with patch("runpod_io.pod_state.runpod_sdk.get_pod", return_value=None):
        assert pod_state.get_pod("missing") is None


def test_format_uptime() -> None:
    assert pod_state.format_uptime(45) == "45s"
    assert pod_state.format_uptime(125) == "2m"
    assert pod_state.format_uptime(3725) == "1h02m"
