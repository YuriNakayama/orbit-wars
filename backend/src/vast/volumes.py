"""Vast.ai network volume helpers (永続化ボリューム検索 / 作成 / 一覧).

uv wheel cache や DVC cache を Vast インスタンス間で永続化するための薄いラッパ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class VolumeOffer:
    """Vast.ai の volume search 結果 1 件 (storage offer)."""

    id: int
    machine_id: int
    disk_space_gb: float
    storage_per_hour: float
    geolocation: str | None
    reliability: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> VolumeOffer:
        return cls(
            id=int(raw.get("ask_contract_id") or raw.get("id", 0)),
            machine_id=int(raw.get("machine_id", 0)),
            disk_space_gb=float(raw.get("disk_space", 0.0)),
            storage_per_hour=float(raw.get("storage_cost", 0.0)),
            geolocation=raw.get("geolocation"),
            reliability=float(raw.get("reliability2", raw.get("reliability", 0.0))),
        )


@dataclass(frozen=True)
class Volume:
    """既存の volume contract."""

    id: int
    name: str
    size_gb: float
    mount_path: str | None
    machine_id: int | None
    storage_per_hour: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Volume:
        return cls(
            id=int(raw.get("id", 0)),
            name=str(raw.get("name") or ""),
            size_gb=float(raw.get("disk_space", raw.get("size", 0.0))),
            mount_path=raw.get("mount_path"),
            machine_id=raw.get("machine_id"),
            storage_per_hour=float(raw.get("storage_cost", 0.0)),
        )


def search_volume_offers(
    sdk: Any,
    *,
    min_size_gb: float = 15.0,
    max_storage_per_hour: float = 0.005,
    min_reliability: float = 0.99,
    limit: int = 10,
) -> list[VolumeOffer]:
    """Volume offer を検索. dph_storage 昇順で limit 件返す."""
    query = (
        f"disk_space>={min_size_gb} "
        f"storage_cost<{max_storage_per_hour} "
        f"reliability>={min_reliability}"
    )
    raw_offers = sdk.search_volumes(
        query=query,
        order="storage_cost",
        limit=limit,
        storage=min_size_gb,
    )
    offers = [VolumeOffer.from_raw(r) for r in raw_offers]
    offers.sort(key=lambda o: o.storage_per_hour)
    return offers


def render_volume_offers(
    offers: list[VolumeOffer], console: Console | None = None
) -> None:
    """Volume offers を rich table で表示."""
    if console is None:
        console = Console()
    table = Table(title="Vast.ai volume offers (cheapest first)")
    table.add_column("#", justify="right")
    table.add_column("offer_id")
    table.add_column("machine_id")
    table.add_column("size_gb", justify="right")
    table.add_column("$/h", justify="right")
    table.add_column("reliability", justify="right")
    table.add_column("region")
    for i, o in enumerate(offers, 1):
        table.add_row(
            str(i),
            str(o.id),
            str(o.machine_id),
            f"{o.disk_space_gb:.0f}",
            f"${o.storage_per_hour:.5f}",
            f"{o.reliability:.3f}",
            o.geolocation or "-",
        )
    console.print(table)


def pick_volume_offer(
    offers: list[VolumeOffer], *, default_index: int = 1, console: Console | None = None
) -> VolumeOffer:
    """Stdin で offer 番号を選択. 空入力 / EOF なら default_index を選ぶ."""
    if console is None:
        console = Console()
    if not offers:
        raise ValueError("no offers available")
    render_volume_offers(offers, console=console)
    try:
        raw = input(f"Pick volume offer # ({default_index}): ").strip()
    except EOFError:
        raw = ""
    if not raw:
        idx = default_index
    else:
        idx = int(raw)
    if idx < 1 or idx > len(offers):
        raise ValueError(f"invalid choice {idx}; expected 1..{len(offers)}")
    return offers[idx - 1]


def create_volume(sdk: Any, *, offer_id: int, size_gb: float, name: str) -> int:
    """Volume を新規作成し volume_id を返す."""
    response = sdk.create_volume(id=offer_id, size=size_gb, name=name)
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"unexpected create_volume response: {type(response).__name__}"
        )
    new_id = response.get("new_contract") or response.get("id")
    if new_id is None:
        raise RuntimeError(f"create_volume response missing id: keys={list(response)}")
    return int(new_id)


def list_volumes(sdk: Any) -> list[Volume]:
    """所有する volumes を一覧."""
    raw = sdk.show_volumes()
    if not isinstance(raw, list):
        return []
    return [Volume.from_raw(r) for r in raw]


def find_volume_by_name(volumes: list[Volume], name: str) -> Volume | None:
    """name 一致 (前方一致 OK) の volume を返す. 複数あれば最新 id."""
    matches = [v for v in volumes if v.name == name]
    if not matches:
        return None
    matches.sort(key=lambda v: v.id, reverse=True)
    return matches[0]


__all__ = [
    "Volume",
    "VolumeOffer",
    "create_volume",
    "find_volume_by_name",
    "list_volumes",
    "pick_volume_offer",
    "render_volume_offers",
    "search_volume_offers",
]
