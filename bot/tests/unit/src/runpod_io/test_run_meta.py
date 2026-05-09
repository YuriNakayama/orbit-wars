"""runpod_io.run_meta は vast.run_meta の re-export + helper のテスト。"""

from __future__ import annotations

from runpod_io.run_meta import (
    SCHEMA_VERSION,
    RunMetadata,
    build_runpod_offer_snapshot,
    generate_run_id,
)


def test_reexports_smoke() -> None:
    assert SCHEMA_VERSION == 1
    meta = RunMetadata()
    assert meta.runpod_pod_id is None
    assert meta.runpod_offer_snapshot is None
    rid = generate_run_id("main", "abc1234deadbeef", 0)
    assert "abc1234" in rid
    assert "seed0" in rid


def test_build_runpod_offer_snapshot() -> None:
    snap = build_runpod_offer_snapshot(
        gpu_type_id="NVIDIA GeForce RTX 3090",
        display_name="RTX 3090",
        memory_gb=24,
        cloud_type="SECURE",
        secure_cloud=True,
        community_cloud=True,
        dph_total=0.43,
        data_center_id="US-KS-2",
    )
    assert snap["gpu_type_id"] == "NVIDIA GeForce RTX 3090"
    assert snap["display_name"] == "RTX 3090"
    assert snap["memory_gb"] == 24
    assert snap["cloud_type"] == "SECURE"
    assert snap["secure_cloud"] is True
    assert snap["community_cloud"] is True
    assert snap["dph_total"] == 0.43
    assert snap["data_center_id"] == "US-KS-2"


def test_build_runpod_offer_snapshot_no_dc() -> None:
    snap = build_runpod_offer_snapshot(
        gpu_type_id="x",
        display_name="x",
        memory_gb=24,
        cloud_type="COMMUNITY",
        secure_cloud=False,
        community_cloud=True,
        dph_total=0.22,
    )
    assert snap["data_center_id"] is None
