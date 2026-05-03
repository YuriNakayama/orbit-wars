"""runpod_io.offers のユニットテスト。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from runpod_io.offers import (
    Offer,
    format_table,
    search_offers,
)


def _stub_sdk(gpus: list[dict[str, Any]], details: dict[str, dict[str, Any]]) -> Any:
    sdk = MagicMock()
    sdk.get_gpus.return_value = gpus
    sdk.get_gpu.side_effect = lambda gpu_id: details.get(gpu_id, {})
    return sdk


def _detail(
    gid: str,
    *,
    memory: int = 24,
    secure: bool = True,
    community: bool = True,
    secure_price: float | None = 0.5,
    community_price: float | None = 0.3,
    secure_spot: float | None = 0.2,
    community_spot: float | None = 0.1,
) -> dict[str, Any]:
    return {
        "id": gid,
        "displayName": gid.replace("NVIDIA ", ""),
        "memoryInGb": memory,
        "secureCloud": secure,
        "communityCloud": community,
        "securePrice": secure_price,
        "communityPrice": community_price,
        "secureSpotPrice": secure_spot,
        "communitySpotPrice": community_spot,
    }


def test_search_offers_secure_only() -> None:
    sdk = _stub_sdk(
        [{"id": "NVIDIA RTX 3090"}, {"id": "NVIDIA RTX 4090"}],
        {
            "NVIDIA RTX 3090": _detail("NVIDIA RTX 3090", secure_price=0.5),
            "NVIDIA RTX 4090": _detail("NVIDIA RTX 4090", secure_price=0.7),
        },
    )
    offers = search_offers(
        sdk,
        gpu_names=["NVIDIA RTX 3090", "NVIDIA RTX 4090"],
        cloud_type="SECURE",
    )
    assert len(offers) == 2
    assert all(o.cloud_type == "SECURE" for o in offers)
    # dph asc sort
    assert offers[0].dph_total == 0.5
    assert offers[1].dph_total == 0.7


def test_search_offers_community_only_filters_secure_only_gpu() -> None:
    sdk = _stub_sdk(
        [{"id": "NVIDIA A100"}, {"id": "NVIDIA RTX 3090"}],
        {
            "NVIDIA A100": _detail(
                "NVIDIA A100", community=False, community_price=None
            ),
            "NVIDIA RTX 3090": _detail("NVIDIA RTX 3090", community_price=0.3),
        },
    )
    offers = search_offers(
        sdk,
        gpu_names=["NVIDIA A100", "NVIDIA RTX 3090"],
        cloud_type="COMMUNITY",
    )
    assert len(offers) == 1
    assert offers[0].gpu_type_id == "NVIDIA RTX 3090"


def test_search_offers_all_emits_two_per_gpu() -> None:
    sdk = _stub_sdk(
        [{"id": "NVIDIA RTX 3090"}],
        {"NVIDIA RTX 3090": _detail("NVIDIA RTX 3090")},
    )
    offers = search_offers(sdk, gpu_names=["NVIDIA RTX 3090"], cloud_type="ALL")
    cloud_types = sorted(o.cloud_type for o in offers)
    assert cloud_types == ["COMMUNITY", "SECURE"]
    # community が cheaper
    assert offers[0].cloud_type == "COMMUNITY"


def test_search_offers_max_dph_filters_expensive() -> None:
    sdk = _stub_sdk(
        [{"id": "NVIDIA A100"}],
        {"NVIDIA A100": _detail("NVIDIA A100", secure_price=2.5)},
    )
    offers = search_offers(
        sdk, gpu_names=["NVIDIA A100"], cloud_type="SECURE", max_dph=1.0
    )
    assert offers == []


def test_search_offers_min_memory_filters_small_gpu() -> None:
    sdk = _stub_sdk(
        [{"id": "NVIDIA T4"}],
        {"NVIDIA T4": _detail("NVIDIA T4", memory=16)},
    )
    offers = search_offers(
        sdk, gpu_names=["NVIDIA T4"], cloud_type="SECURE", min_memory_gb=24
    )
    assert offers == []


def test_search_offers_limit() -> None:
    gpus = [{"id": f"GPU-{i}"} for i in range(20)]
    details = {
        f"GPU-{i}": _detail(f"GPU-{i}", secure_price=float(i)) for i in range(20)
    }
    sdk = _stub_sdk(gpus, details)
    offers = search_offers(
        sdk,
        gpu_names=[f"GPU-{i}" for i in range(20)],
        cloud_type="SECURE",
        max_dph=100.0,
        limit=5,
    )
    assert len(offers) == 5
    assert offers[0].dph_total == 0.0
    assert offers[-1].dph_total == 4.0


def test_search_offers_invalid_cloud_type() -> None:
    sdk = _stub_sdk([], {})
    with pytest.raises(ValueError):
        search_offers(sdk, gpu_names=["x"], cloud_type="WRONG")


def test_offer_to_snapshot_round_trip() -> None:
    offer = Offer(
        gpu_type_id="NVIDIA RTX 3090",
        display_name="RTX 3090",
        memory_gb=24,
        secure_cloud=True,
        community_cloud=True,
        secure_price=0.5,
        community_price=0.3,
        secure_spot_price=0.2,
        community_spot_price=0.1,
        cloud_type="SECURE",
        dph_total=0.5,
    )
    snap = offer.to_snapshot()
    assert snap["gpu_type_id"] == "NVIDIA RTX 3090"
    assert snap["cloud_type"] == "SECURE"
    assert snap["dph_total"] == 0.5


def test_format_table_returns_table() -> None:
    offer = Offer(
        gpu_type_id="x",
        display_name="x",
        memory_gb=24,
        secure_cloud=True,
        community_cloud=True,
        secure_price=0.5,
        community_price=0.3,
        secure_spot_price=None,
        community_spot_price=None,
        cloud_type="SECURE",
        dph_total=0.5,
    )
    table = format_table([offer])
    assert table.row_count == 1
