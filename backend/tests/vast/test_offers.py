"""offers.py のユニットテスト。SDK は MagicMock で stub。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vast.offers import (
    Offer,
    _build_query,
    _parse_offer,
    format_table,
    pick_offer,
    search_offers,
)


def _make_raw(
    offer_id: int,
    *,
    dph: float,
    gpu: str = "RTX_3090",
    reliability: float = 0.995,
) -> dict[str, object]:
    return {
        "id": offer_id,
        "gpu_name": gpu,
        "num_gpus": 1,
        "dph_total": dph,
        "reliability2": reliability,
        "geolocation": "US",
        "cuda_max_good": 12.4,
        "inet_down": 500.0,
        "verified": True,
    }


def test_build_query_uses_in_list_for_multiple_gpus() -> None:
    q = _build_query(
        gpu_names=("RTX_3090", "RTX_4090"),
        max_dph=0.5,
        min_reliability=0.99,
        min_cuda=12.0,
    )
    assert "gpu_name in [RTX_3090,RTX_4090]" in q
    assert "num_gpus=1" in q
    assert "reliability>=0.99" in q
    assert "cuda_max_good>=12.0" in q
    assert "dph_total<0.5" in q
    assert "rentable=true" in q
    assert "verified=true" in q


def test_build_query_uses_eq_for_single_gpu() -> None:
    q = _build_query(
        gpu_names=("RTX_3090",),
        max_dph=0.5,
        min_reliability=0.99,
        min_cuda=12.0,
    )
    assert "gpu_name=RTX_3090" in q
    assert "in [" not in q


def test_parse_offer_basic() -> None:
    raw = _make_raw(123, dph=0.13)
    offer = _parse_offer(raw)
    assert offer.id == 123
    assert offer.gpu_name == "RTX_3090"
    assert offer.dph_total == pytest.approx(0.13)
    assert offer.reliability == pytest.approx(0.995)
    assert offer.geolocation == "US"
    assert offer.verified is True


def test_parse_offer_falls_back_to_reliability_field() -> None:
    raw = _make_raw(1, dph=0.10)
    raw.pop("reliability2")
    raw["reliability"] = 0.93
    offer = _parse_offer(raw)
    assert offer.reliability == pytest.approx(0.93)


def test_search_offers_sorts_by_dph_and_truncates() -> None:
    sdk = MagicMock()
    sdk.search_offers.return_value = [
        _make_raw(3, dph=0.30),
        _make_raw(1, dph=0.10),
        _make_raw(2, dph=0.20),
    ]
    offers = search_offers(sdk, limit=2)
    assert [o.id for o in offers] == [1, 2]
    assert sdk.search_offers.call_count == 1
    kwargs = sdk.search_offers.call_args.kwargs
    assert kwargs["type"] == "on-demand"
    assert kwargs["order"] == "dph_total"


def test_search_offers_unexpected_response_raises() -> None:
    sdk = MagicMock()
    sdk.search_offers.return_value = "not a list"
    with pytest.raises(RuntimeError, match="unexpected"):
        search_offers(sdk)


def test_format_table_renders_rows() -> None:
    offers = [
        Offer(
            id=1,
            gpu_name="RTX_3090",
            num_gpus=1,
            dph_total=0.13,
            reliability=0.995,
            geolocation="US",
            cuda_max_good=12.4,
            inet_down=500.0,
            verified=True,
        )
    ]
    table = format_table(offers)
    assert table.row_count == 1
    assert "offer_id" in {col.header for col in table.columns}


def test_pick_offer_within_range(monkeypatch: pytest.MonkeyPatch) -> None:
    offers = [
        Offer(1, "RTX_3090", 1, 0.13, 0.995, "US", 12.4, 500.0, True),
        Offer(2, "RTX_4090", 1, 0.29, 0.995, "EU", 12.4, 800.0, True),
    ]
    monkeypatch.setattr("vast.offers.IntPrompt.ask", lambda *_a, **_k: 2)
    chosen = pick_offer(offers)
    assert chosen.id == 2


def test_pick_offer_empty_raises() -> None:
    with pytest.raises(RuntimeError, match="no offers"):
        pick_offer([])


def test_pick_offer_retries_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    offers = [Offer(1, "RTX_3090", 1, 0.13, 0.995, "US", 12.4, 500.0, True)]
    answers = iter([5, 1])

    monkeypatch.setattr("vast.offers.IntPrompt.ask", lambda *_a, **_k: next(answers))
    chosen = pick_offer(offers)
    assert chosen.id == 1
