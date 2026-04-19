"""Replay → parquet preprocess for case3 IL baseline.

Pipeline:
  1) load index.parquet, filter mode ∈ cfg.modes, drop draws.
  2) per (match_id, player) pick winner side only; require both seats' rating_mu
     to clear the rating_quantile cutoff (top 25% by default).
  3) split episodes 90/10 deterministically by hash(match_id).
  4) walk steps[*][player_slot]; for each (obs, action) emit one row per action,
     plus one no-op row when the action list is empty.
  5) write data/mart/case3/{train,val}.parquet via polars.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
import yaml

from pipeline.case3.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize,
)

logger = logging.getLogger(__name__)

NO_OP_LABEL = MAX_PLANETS  # virtual slot for "do nothing"


@dataclass(frozen=True)
class PreprocessReport:
    rating_cutoff: float
    episodes_total: int
    episodes_kept: int
    train_frames: int
    val_frames: int
    out_train: Path
    out_val: Path


def _fleet_target_planet_id(
    src_id: int,
    src_x: float,
    src_y: float,
    angle: float,
    ships: int,
    planets: list[list[Any]],
) -> int | None:
    """Mirror world_model.fleet_target_planet but return planet id only.

    Excludes the source planet from candidates (fleets spawn just outside its
    radius in the real sim, so self-hit is a reconstruction artifact).
    """
    speed = max(0.5, 2.0 - 0.05 * math.sqrt(max(1, ships)))
    if speed <= 0:
        return None
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    best_id: int | None = None
    best_t = 1e9
    for row in planets:
        pid, _owner, px, py, radius, _s, _p = row
        if int(pid) == src_id:
            continue
        dx = float(px) - src_x
        dy = float(py) - src_y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = float(radius) * float(radius)
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        t = hit_d / speed
        if t < best_t:
            best_t = t
            best_id = int(pid)
    return best_id


def _ships_bucket(ships: int, src_ships: int, num_buckets: int) -> int:
    if src_ships <= 0:
        return 0
    ratio = max(0.0, min(1.0, ships / max(1, src_ships)))
    bucket = int(ratio * num_buckets)
    return min(bucket, num_buckets - 1)


def _split_episode(match_id: str, val_split: float) -> str:
    h = hashlib.sha1(match_id.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], "big") / 2**32
    return "val" if val < val_split else "train"


def _winner_slots(row: dict[str, Any]) -> list[int]:
    if row.get("draw"):
        return []
    winner = int(row.get("winner", -1))
    if winner < 0:
        return []
    return [winner]


def _build_frames(
    obs: dict[str, Any],
    action_list: list[list[Any]],
    src_planets: list[list[Any]],
    ships_buckets: int,
) -> list[dict[str, Any]]:
    """Build one parquet row per action (or one no-op row if empty)."""
    batch, snap = featurize(obs)
    planet_feats = batch.planet_feats[0].numpy().astype(np.float32)
    global_feats = batch.global_feats[0].numpy().astype(np.float32)
    planet_mask = batch.planet_mask[0].numpy().astype(np.bool_)
    my_planet_mask = batch.my_planet_mask[0].numpy().astype(np.bool_)
    target_mask = batch.target_mask[0].numpy().astype(np.bool_)

    pid_to_slot = {pid: i for i, pid in enumerate(snap.planet_ids)}
    src_ships_by_id = {int(p[0]): int(p[5]) for p in src_planets}

    base = {
        "planet_feats": planet_feats.reshape(-1).tolist(),
        "global_feats": global_feats.tolist(),
        "planet_mask": planet_mask.tolist(),
        "my_planet_mask": my_planet_mask.tolist(),
        "target_mask": target_mask.tolist(),
    }

    if not action_list:
        row = dict(base)
        row.update(
            from_label=NO_OP_LABEL,
            target_label=NO_OP_LABEL,
            ships_label=0,
            is_noop=True,
        )
        return [row]

    rows: list[dict[str, Any]] = []
    for act in action_list:
        from_pid, angle, ships = int(act[0]), float(act[1]), int(act[2])
        from_slot = pid_to_slot.get(from_pid)
        if from_slot is None:
            continue
        target_pid = _fleet_target_planet_id(
            from_pid,
            float(src_planets[from_slot][2]),
            float(src_planets[from_slot][3]),
            angle,
            ships,
            src_planets,
        )
        target_slot = pid_to_slot.get(target_pid) if target_pid is not None else None
        if target_slot is None:
            target_slot = NO_OP_LABEL
        bucket = _ships_bucket(ships, src_ships_by_id.get(from_pid, 0), ships_buckets)
        row = dict(base)
        row.update(
            from_label=int(from_slot),
            target_label=int(target_slot),
            ships_label=int(bucket),
            is_noop=False,
        )
        rows.append(row)
    return rows


def _iter_episode_frames(
    replay_path: Path,
    player_slots: list[int],
    ships_buckets: int,
) -> list[dict[str, Any]]:
    with gzip.open(replay_path, "rt") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    out: list[dict[str, Any]] = []
    for step in steps:
        for slot in player_slots:
            if slot >= len(step):
                continue
            sd = step[slot]
            obs = sd.get("observation") or {}
            if not obs:
                continue
            action_list = sd.get("action") or []
            planets = obs.get("planets") or []
            if not planets:
                continue
            frames = _build_frames(obs, action_list, planets, ships_buckets)
            out.extend(frames)
    return out


def _filter_index(
    index: pl.DataFrame,
    modes: list[str],
    rating_quantile: float,
) -> tuple[pl.DataFrame, float]:
    df = index.filter(pl.col("mode").is_in(modes))
    df = df.filter(~pl.col("draw"))
    rating_cols = [c for c in df.columns if c.endswith("_rating_mu")]
    rating = pl.concat([df.select(pl.col(c).alias("mu")) for c in rating_cols]).filter(
        pl.col("mu") > 0
    )
    if rating.is_empty():
        return df, 0.0
    cutoff = float(rating.quantile(rating_quantile).item())

    # require winner side rating to clear cutoff. We only know the winner slot,
    # so build per-row check via list of (winner_col -> rating_col) lookup.
    cond = pl.lit(False)
    for slot, col in enumerate(rating_cols):
        cond = cond | ((pl.col("winner") == slot) & (pl.col(col) >= cutoff))
    df = df.filter(cond)
    return df, cutoff


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write empty schema-compatible file
        empty = pl.DataFrame(schema=_empty_schema())
        empty.write_parquet(path)
        return 0
    df = pl.DataFrame(rows)
    df.write_parquet(path)
    return df.height


def _empty_schema() -> dict[str, pl.DataType]:
    return {
        "planet_feats": pl.List(pl.Float32),
        "global_feats": pl.List(pl.Float32),
        "planet_mask": pl.List(pl.Boolean),
        "my_planet_mask": pl.List(pl.Boolean),
        "target_mask": pl.List(pl.Boolean),
        "from_label": pl.Int32(),
        "target_label": pl.Int32(),
        "ships_label": pl.Int32(),
        "is_noop": pl.Boolean(),
    }


def preprocess(cfg: dict[str, Any]) -> PreprocessReport:
    data_cfg = cfg["data"]
    modes = list(data_cfg.get("modes", ["1v1"]))
    rating_quantile = float(data_cfg.get("rating_quantile", 0.75))
    val_split = float(data_cfg.get("val_split", 0.10))
    ships_buckets = int(cfg.get("model", {}).get("ships_buckets", 5))
    max_episodes = data_cfg.get("max_episodes")
    out_train = Path(data_cfg["out_train"])
    out_val = Path(data_cfg["out_val"])
    index_path = Path(data_cfg["kaggle_index_root"])
    # replay_path stored in index is relative to index.parquet's parent dir
    base_dir = index_path.parent

    index = pl.read_parquet(index_path)
    filtered, cutoff = _filter_index(index, modes, rating_quantile)
    episodes_total = filtered.height
    if max_episodes is not None:
        filtered = filtered.head(int(max_episodes))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rows = filtered.to_dicts()
    expected_dim = MAX_PLANETS * PLANET_FEAT_DIM
    assert expected_dim > 0 and GLOBAL_FEAT_DIM > 0  # sanity

    kept = 0
    for rec in rows:
        slots = _winner_slots(rec)
        if not slots:
            continue
        rp = rec.get("replay_path") or ""
        replay_path = (base_dir / rp).resolve() if rp else None
        if replay_path is None or not replay_path.exists():
            continue
        frames = _iter_episode_frames(replay_path, slots, ships_buckets)
        if not frames:
            continue
        bucket = _split_episode(str(rec["match_id"]), val_split)
        target = val_rows if bucket == "val" else train_rows
        target.extend(frames)
        kept += 1

    n_train = _write_parquet(train_rows, out_train)
    n_val = _write_parquet(val_rows, out_val)

    report = PreprocessReport(
        rating_cutoff=cutoff,
        episodes_total=episodes_total,
        episodes_kept=kept,
        train_frames=n_train,
        val_frames=n_val,
        out_train=out_train,
        out_val=out_val,
    )
    logger.info(
        "preprocess done: rating_cutoff=%.2f episodes=%d/%d frames train=%d val=%d",
        report.rating_cutoff,
        report.episodes_kept,
        report.episodes_total,
        report.train_frames,
        report.val_frames,
    )
    return report


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(  # noqa: B008
        Path("pipeline/case3/configs/il_baseline.yaml"),
        "--config",
        "-c",
        help="YAML config path",
    ),
) -> None:
    """CLI: run preprocess with the given config."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with config.open() as f:
        cfg = yaml.safe_load(f)
    report = preprocess(cfg)
    typer.echo(
        f"rating_cutoff={report.rating_cutoff:.2f} "
        f"episodes_kept={report.episodes_kept}/{report.episodes_total} "
        f"train_frames={report.train_frames} val_frames={report.val_frames}"
    )


if __name__ == "__main__":
    app()
