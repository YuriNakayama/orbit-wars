"""Shared fixtures for RunPod unit tests."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def sample_pod_payload() -> dict[str, Any]:
    return {
        "id": "pod-abc",
        "name": "my-run",
        "desiredStatus": "RUNNING",
        "costPerHr": 0.45,
        "uptimeSeconds": 3725,
        "imageName": "pytorch/pytorch:2.0",
        "machine": {"gpuDisplayName": "RTX 3090"},
    }
