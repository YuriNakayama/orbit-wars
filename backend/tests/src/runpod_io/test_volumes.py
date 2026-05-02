"""runpod_io.volumes のユニットテスト。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from runpod_io.volumes import (
    KNOWN_DATA_CENTERS,
    Volume,
    create_volume,
    find_volume_by_name,
    list_volumes,
    search_volume_offers,
    validate_volume_name,
)


def _stub_graphql(response: Any) -> Any:
    sdk = MagicMock()
    sdk.run_graphql_query.return_value = response
    return sdk


def test_list_volumes_parses_payload() -> None:
    sdk = _stub_graphql(
        {
            "data": {
                "myself": {
                    "networkVolumes": [
                        {
                            "id": "vol-1",
                            "name": "orbit_wars_cache_runpod",
                            "size": 15,
                            "dataCenterId": "US-KS-2",
                        },
                        {
                            "id": "vol-2",
                            "name": "other",
                            "size": 50,
                            "dataCenterId": "EU-RO-1",
                        },
                    ]
                }
            }
        }
    )
    volumes = list_volumes(sdk)
    assert len(volumes) == 2
    assert volumes[0].id == "vol-1"
    assert volumes[0].name == "orbit_wars_cache_runpod"
    assert volumes[0].size_gb == 15
    assert volumes[0].data_center_id == "US-KS-2"


def test_list_volumes_empty_data() -> None:
    sdk = _stub_graphql({"data": {"myself": {"networkVolumes": []}}})
    assert list_volumes(sdk) == []


def test_list_volumes_missing_keys() -> None:
    sdk = _stub_graphql({"data": {}})
    assert list_volumes(sdk) == []


def test_search_volume_offers_returns_known_dcs() -> None:
    offers = search_volume_offers(min_size_gb=15)
    assert len(offers) == len(KNOWN_DATA_CENTERS)
    assert all(o.suggested_size_gb == 15 for o in offers)


def test_search_volume_offers_filter_by_dc() -> None:
    offers = search_volume_offers(min_size_gb=15, data_center_id="US-KS-2")
    assert len(offers) == 1
    assert offers[0].data_center_id == "US-KS-2"


def test_search_volume_offers_unknown_dc_returns_empty() -> None:
    offers = search_volume_offers(min_size_gb=15, data_center_id="MARS-1")
    assert offers == []


def test_create_volume_returns_id() -> None:
    sdk = _stub_graphql(
        {
            "data": {
                "createNetworkVolume": {
                    "id": "vol-new",
                    "name": "abc",
                    "size": 15,
                    "dataCenterId": "US-KS-2",
                }
            }
        }
    )
    new_id = create_volume(sdk, name="abc", size_gb=15, data_center_id="US-KS-2")
    assert new_id == "vol-new"
    args, kwargs = sdk.run_graphql_query.call_args
    assert "createNetworkVolume" in args[0]
    assert kwargs["variables"]["input"]["name"] == "abc"
    assert kwargs["variables"]["input"]["size"] == 15
    assert kwargs["variables"]["input"]["dataCenterId"] == "US-KS-2"


def test_create_volume_invalid_dc() -> None:
    sdk = _stub_graphql({})
    with pytest.raises(ValueError, match="unknown data_center_id"):
        create_volume(sdk, name="abc", size_gb=15, data_center_id="FAKE-1")


def test_create_volume_invalid_name() -> None:
    sdk = _stub_graphql({})
    with pytest.raises(ValueError):
        create_volume(sdk, name="bad name!", size_gb=15, data_center_id="US-KS-2")


def test_create_volume_missing_id_in_response() -> None:
    sdk = _stub_graphql({"data": {"createNetworkVolume": {}}})
    with pytest.raises(RuntimeError, match="returned no id"):
        create_volume(sdk, name="abc", size_gb=15, data_center_id="US-KS-2")


def test_validate_volume_name_accepts_safe_chars() -> None:
    validate_volume_name("orbit_wars_cache_runpod")
    validate_volume_name("abc-xyz-123")


def test_validate_volume_name_rejects_special_chars() -> None:
    for bad in ["spaces here", "name!", "../escape", "x" * 65]:
        with pytest.raises(ValueError):
            validate_volume_name(bad)


def test_find_volume_by_name() -> None:
    volumes = [
        Volume(id="v1", name="abc", size_gb=15, data_center_id="US-KS-2"),
        Volume(id="v2", name="other", size_gb=50, data_center_id="EU-RO-1"),
        Volume(id="v3", name="abc", size_gb=20, data_center_id="EU-NL-1"),
    ]
    found = find_volume_by_name(volumes, "abc")
    assert found is not None
    assert found.id == "v3"  # last match


def test_find_volume_by_name_not_found() -> None:
    volumes = [Volume(id="v1", name="x", size_gb=10, data_center_id="US-KS-2")]
    assert find_volume_by_name(volumes, "missing") is None
