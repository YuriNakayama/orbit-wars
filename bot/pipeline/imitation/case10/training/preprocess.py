"""Replay → parquet preprocess for imitation/case10.

Per-frame supervision schema:

  - planet_feats : (MAX_PLANETS * 35,) float32
  - global_feats : (20,)              float32
  - planet_mask  : (MAX_PLANETS,)     bool
  - my_planet_mask, target_mask : (MAX_PLANETS,) bool
  - candidate_feats : (MAX_PLANETS * CAND_K * CAND_FEAT_DIM,) float32
  - candidate_mask  : (MAX_PLANETS * CAND_K,) bool
  - candidate_pid   : (MAX_PLANETS * CAND_K,) int32
  - cand_slot_per_src : (MAX_PLANETS,) int32  (-1 = src not active in label)
                        0 = no-op label (src had option to fire but did not)
                        1..K-1 = candidate slot the actual fired target maps to
  - is_noop : bool

For "fired" sources: reverse-resolve the (angle, ships) to a target planet id;
look up that pid in candidate_pid; if found at slot s, set cand_slot_per_src=s;
if not present in candidates (rare, K=8 might not include the actual target),
set cand_slot_per_src=-1 (excluded from loss).

For "not-fired my planets": cand_slot_per_src=0 (no-op label).
For "not my planets": cand_slot_per_src=-1.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

from pipeline.imitation.case10.policy import featurizer
from pipeline.imitation.case10.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case10.policy.featurizer import (
    MAX_PLANETS,
    HistoryState,
)
from pipeline.imitation.case10.policy.geometry import Planet, aim_with_prediction
from pipeline.imitation.case10.policy.templates import (
    T_NO_OP,
    classify_actual_target,
)

logger = logging.getLogger(__name__)

UNUSED_LABEL = -1
ANGLE_TOLERANCE = 0.20  # radians (~11.5deg) — angle reverse-resolve match window
PROGRESS_LOG_EVERY = 100  # log every N episodes processed (kept or skipped)
FLUSH_EVERY_FRAMES = 5000  # flush parquet rows when buffer reaches this size
SHIPS_BUCKETS = 4  # 0:25%, 1:50%, 2:75%, 3:100% (case7 流)


def _ships_bucket(ships: int, src_ships: int) -> int:
    if src_ships <= 0:
        return 0
    ratio = max(0.0, min(1.0, ships / max(1, src_ships)))
    bucket = int(round(ratio * (SHIPS_BUCKETS - 1) - 0.001))
    return max(0, min(SHIPS_BUCKETS - 1, bucket))


@dataclass(frozen=True)
class PreprocessReport:
    rating_cutoff: float
    episodes_total: int
    episodes_kept: int
    train_frames: int
    val_frames: int
    out_train: Path
    out_val: Path
    label_outside_candidates: int
    fired_total: int


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bot").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def _load_replay_json(replay_path: Path) -> dict[str, Any]:
    with gzip.open(replay_path, "rb") as f:
        return orjson.loads(f.read())


def _build_planet(row: list[Any]) -> Planet:
    return Planet(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)


def _resolve_action_target(
    src: Planet,
    angle: float,
    ships: int,
    candidates: list[Planet],
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> int | None:
    best_id: int | None = None
    best_diff = ANGLE_TOLERANCE
    for cand in candidates:
        if cand.id == src.id:
            continue
        aim = aim_with_prediction(
            src, cand, ships, initial_by_id, ang_vel, comets, comet_ids
        )
        if aim is None:
            continue
        diff = _angle_diff(aim[0], angle)
        if diff < best_diff:
            best_diff = diff
            best_id = cand.id
    return best_id


def _split_episode(match_id: str, val_split: float) -> str:
    h = hashlib.sha1(match_id.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], "big") / 2**32
    return "val" if val < val_split else "train"


def _player_slots(row: dict[str, Any], num_slots: int) -> list[int]:
    if row.get("draw"):
        return []
    return list(range(num_slots))


def _swap_owner_01(owner: int) -> int:
    if owner == 0:
        return 1
    if owner == 1:
        return 0
    return owner


def _swap_obs_players(obs: dict[str, Any], winner_slot: int) -> dict[str, Any]:
    """Return a new obs where player 0 / player 1 ownership is swapped.

    Used by loser_swap=true: the original loser-slot frame is rewritten so the
    winner appears as ``obs.player = winner_slot`` while the board state is
    relabeled accordingly. Planets/fleets owned by 0 become 1 and vice versa;
    neutral (-1) is preserved.
    """
    new_planets: list[list[Any]] = []
    for row in obs.get("planets") or []:
        nr = list(row)
        nr[1] = _swap_owner_01(int(nr[1]))
        new_planets.append(nr)
    new_fleets: list[list[Any]] = []
    for row in obs.get("fleets") or []:
        nr = list(row)
        nr[1] = _swap_owner_01(int(nr[1]))
        new_fleets.append(nr)
    new_initial: list[list[Any]] = []
    for row in obs.get("initial_planets") or []:
        nr = list(row)
        nr[1] = _swap_owner_01(int(nr[1]))
        new_initial.append(nr)
    return {
        **obs,
        "player": winner_slot,
        "planets": new_planets,
        "fleets": new_fleets,
        "initial_planets": new_initial,
    }


def _episode_meta_fields(rec: dict[str, Any]) -> dict[str, Any]:
    """Static (per-episode) fields written into the side index.parquet."""
    turn_count = rec.get("turns")
    if turn_count is None:
        turn_count = -1
    p0_sub = rec.get("agent_0_submission_id")
    p1_sub = rec.get("agent_1_submission_id")
    p0_mu = rec.get("agent_0_rating_mu")
    p1_mu = rec.get("agent_1_rating_mu")
    return {
        "turn_count": int(turn_count),
        "mode": str(rec.get("mode", "")),
        "p0_submission_id": str(p0_sub) if p0_sub is not None else "",
        "p1_submission_id": str(p1_sub) if p1_sub is not None else "",
        "p0_rating_mu": float(p0_mu) if p0_mu is not None else 0.0,
        "p1_rating_mu": float(p1_mu) if p1_mu is not None else 0.0,
    }


def _swap_action_target(action: list[Any]) -> list[Any]:
    """Action shape is [from_pid, angle, ships] — planet ids are stable across
    the swap (only ownership labels change), so the action list is unchanged.
    """
    return list(action)


def _build_frame(
    obs: dict[str, Any],
    action_list: list[list[Any]],
    raw_planets: list[list[Any]],
    history: HistoryState,
) -> tuple[dict[str, Any], int, int] | None:
    """Build one parquet row + (fired_total_delta, outside_candidates_delta)."""
    batch, snap = featurizer.featurize(obs, history)
    planet_feats = batch.planet_feats[0].numpy().astype(np.float32)
    global_feats = batch.global_feats[0].numpy().astype(np.float32)
    planet_mask = batch.planet_mask[0].numpy().astype(np.bool_)
    my_planet_mask = batch.my_planet_mask[0].numpy().astype(np.bool_)
    target_mask = batch.target_mask[0].numpy().astype(np.bool_)
    cand_feats = batch.candidate_feats[0].numpy().astype(np.float32)
    cand_mask = batch.candidate_mask[0].numpy().astype(np.bool_)
    cand_pid = batch.candidate_pid[0].numpy().astype(np.int32)
    template_ctx = batch.template_ctx[0].numpy().astype(np.float32)
    player = int(obs.get("player", 0) or 0)

    pid_to_slot = {pid: i for i, pid in enumerate(snap.planet_ids)}
    planets = [_build_planet(row) for row in raw_planets]
    by_id = {p.id: p for p in planets}
    raw_by_id = {int(row[0]): row for row in raw_planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    cand_slot_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)
    ship_label_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)
    # 3-head labels (case7 流): from_multihot / target_per_src (template id) /
    # ships_per_src (4-bucket).
    from_multihot = np.zeros(MAX_PLANETS, dtype=np.bool_)
    target_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)
    ships_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)
    # candidate_ships variant 用: ships を 4-bucket 化 (cand_slot fired のみ)。
    ships_bucket_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)

    # Default: my_planets that did not fire → label = 0 (no-op slot)
    for slot in range(MAX_PLANETS):
        if my_planet_mask[slot]:
            cand_slot_per_src[slot] = 0
            target_per_src[slot] = T_NO_OP

    fired_total = 0
    outside = 0

    for act in action_list:
        from_pid = int(act[0])
        angle = float(act[1])
        ships = int(act[2])
        maybe_slot = pid_to_slot.get(from_pid)
        if maybe_slot is None or not my_planet_mask[maybe_slot]:
            continue
        slot = int(maybe_slot)
        src = by_id.get(from_pid)
        if src is None:
            continue
        target_pid = _resolve_action_target(
            src, angle, ships, planets, initial_by_id, ang_vel, comets, comet_ids
        )
        fired_total += 1
        # 3-head label: from は fired を multihot、target は template 分類、
        # ships は src.ships に対する比率の bucket。
        from_multihot[slot] = True
        ships_per_src[slot] = int(_ships_bucket(ships, src.ships))
        if target_pid is None:
            target_template = T_NO_OP
            cand_slot_per_src[slot] = UNUSED_LABEL
            outside += 1
        else:
            tgt_row = raw_by_id.get(int(target_pid))
            if tgt_row is None:
                target_template = T_NO_OP
            else:
                target_template = int(
                    classify_actual_target(
                        list(raw_planets[slot]), tgt_row, raw_planets, player
                    )
                )
            # candidate label: K=8 内に target_pid があれば slot_idx、なければ UNUSED。
            slot_idx = -1
            for k in range(1, CAND_K):
                if int(cand_pid[slot, k]) == int(target_pid):
                    slot_idx = k
                    break
            if slot_idx == -1:
                cand_slot_per_src[slot] = UNUSED_LABEL
                outside += 1
            else:
                cand_slot_per_src[slot] = slot_idx
                ship_label_per_src[slot] = ships
                ships_bucket_per_src[slot] = int(_ships_bucket(ships, src.ships))
        target_per_src[slot] = int(target_template)

    is_noop = bool(np.all(cand_slot_per_src[my_planet_mask] == 0))

    row = {
        "planet_feats": planet_feats.reshape(-1).tolist(),
        "global_feats": global_feats.tolist(),
        "planet_mask": planet_mask.tolist(),
        "my_planet_mask": my_planet_mask.tolist(),
        "target_mask": target_mask.tolist(),
        "template_ctx": template_ctx.reshape(-1).tolist(),
        "candidate_feats": cand_feats.reshape(-1).tolist(),
        "candidate_mask": cand_mask.reshape(-1).tolist(),
        "candidate_pid": cand_pid.reshape(-1).tolist(),
        "cand_slot_per_src": cand_slot_per_src.tolist(),
        "ship_label_per_src": ship_label_per_src.tolist(),
        "ships_bucket_per_src": ships_bucket_per_src.tolist(),
        "from_multihot": from_multihot.tolist(),
        "target_per_src": target_per_src.tolist(),
        "ships_per_src": ships_per_src.tolist(),
        "is_noop": is_noop,
    }
    return row, fired_total, outside


def _iter_episode_frames(
    replay_path: Path,
    player_slots: list[int],
    *,
    loser_swap: bool = False,
    winner_slot: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Iterate replay frames and emit (frame_rows, frame_metas, fired, outside).

    With ``loser_swap=true`` and a known ``winner_slot``, the loser slot's frame
    is rewritten via :func:`_swap_obs_players` so the board is relabeled to the
    winner's perspective and the winner's action at the same step is used as
    the supervision label. Frame count is unchanged (1 ep -> 2 samples).

    The returned ``frame_metas`` list is row-aligned with ``frame_rows`` and
    captures (player_slot, source_slot, is_loser_view, step_idx).
    """
    data = _load_replay_json(replay_path)
    steps = data.get("steps", [])
    out_rows: list[dict[str, Any]] = []
    out_metas: list[dict[str, Any]] = []
    histories: dict[int, HistoryState] = {slot: HistoryState() for slot in player_slots}
    fired_total = 0
    outside_total = 0
    apply_swap = loser_swap and winner_slot is not None

    for step_idx, step in enumerate(steps):
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
            if obs.get("step") is None:
                obs = {**obs, "step": step_idx}
            if obs.get("player") is None:
                obs = {**obs, "player": slot}

            is_loser_view = False
            view_slot = slot
            if apply_swap and slot != winner_slot:
                # Loser slot frame: swap board ownership and use winner's
                # action so the model learns counterfactual "what the winner
                # would have done from this losing position".
                obs = _swap_obs_players(obs, winner_slot)
                planets = obs.get("planets") or []
                wd = step[winner_slot] if winner_slot < len(step) else {}
                action_list = _swap_action_target(wd.get("action") or [])
                is_loser_view = True
                view_slot = winner_slot

            history = histories[slot]
            built = _build_frame(obs, action_list, planets, history)
            if built is not None:
                row, fired, outside = built
                out_rows.append(row)
                out_metas.append(
                    {
                        "player_slot": int(view_slot),
                        "source_slot": int(slot),
                        "is_loser_view": bool(is_loser_view),
                        "step_idx": int(step_idx),
                    }
                )
                fired_total += fired
                outside_total += outside
            featurizer.update_history(history, obs, action_list)
    return out_rows, out_metas, fired_total, outside_total


@dataclass(frozen=True)
class _EpisodeResult:
    bucket: str  # "train" | "val"
    frames: list[dict[str, Any]]
    metas: list[dict[str, Any]]  # row-aligned with `frames`
    match_id: str
    winner_slot: int  # -1 when unknown / draw
    fired: int
    outside: int
    skip_reason: str | None  # None | "no_slots" | "no_replay" | "no_frames"


def _process_episode(
    rec: dict[str, Any],
    base_dir: Path,
    val_split: float,
    num_player_slots: int,
    *,
    loser_swap: bool = False,
) -> _EpisodeResult:
    """Worker: process one episode record into frames.

    Pure function (importable / picklable). Returns a single result.
    """
    match_id = str(rec.get("match_id", ""))
    winner_raw = rec.get("winner")
    winner_slot = int(winner_raw) if winner_raw is not None else -1
    if num_player_slots == 0 or rec.get("draw"):
        return _EpisodeResult(
            bucket="",
            frames=[],
            metas=[],
            match_id=match_id,
            winner_slot=winner_slot,
            fired=0,
            outside=0,
            skip_reason="no_slots",
        )
    slots = list(range(num_player_slots))
    rp = rec.get("replay_path") or ""
    replay_path = (base_dir / rp).resolve() if rp else None
    if replay_path is None or not replay_path.exists():
        return _EpisodeResult(
            bucket="",
            frames=[],
            metas=[],
            match_id=match_id,
            winner_slot=winner_slot,
            fired=0,
            outside=0,
            skip_reason="no_replay",
        )
    swap_arg = loser_swap and winner_slot in (0, 1)
    frames, metas, fired, outside = _iter_episode_frames(
        replay_path,
        slots,
        loser_swap=swap_arg,
        winner_slot=winner_slot if swap_arg else None,
    )
    if not frames:
        return _EpisodeResult(
            bucket="",
            frames=[],
            metas=[],
            match_id=match_id,
            winner_slot=winner_slot,
            fired=fired,
            outside=outside,
            skip_reason="no_frames",
        )
    bucket = _split_episode(match_id, val_split)
    return _EpisodeResult(
        bucket=bucket,
        frames=frames,
        metas=metas,
        match_id=match_id,
        winner_slot=winner_slot,
        fired=fired,
        outside=outside,
        skip_reason=None,
    )


def _filter_index(
    index: pl.DataFrame,
    modes: list[str],
    rating_quantile: float,
    *,
    base_dir: Path,
    top_submission_limit: int | None = None,
    turn_min: int | None = None,
    turn_max: int | None = None,
) -> tuple[pl.DataFrame, float]:
    df = index.filter(pl.col("mode").is_in(modes))
    df = df.filter(~pl.col("draw"))
    rating_cols = [c for c in df.columns if c.endswith("_rating_mu")]
    submission_cols = [c for c in df.columns if c.endswith("_submission_id")]
    if top_submission_limit is not None:
        parts: list[pl.DataFrame] = []
        for mu_col in rating_cols:
            prefix = mu_col[: -len("_rating_mu")]
            submission_col = f"{prefix}_submission_id"
            if submission_col not in df.columns:
                continue
            parts.append(
                df.select(
                    pl.col(submission_col).alias("submission_id"),
                    pl.col(mu_col).alias("mu"),
                )
            )
        ratings = pl.concat(parts).filter(pl.col("mu") > 0)
        if ratings.is_empty():
            return df, 0.0
        ranked = (
            ratings.group_by("submission_id")
            .agg(pl.col("mu").max().alias("max_mu"))
            .sort("max_mu", descending=True)
        )
        top = ranked.head(int(top_submission_limit))
        cutoff = float(top["max_mu"][-1])
        top_ids = top["submission_id"].to_list()
        cond = pl.lit(False)
        for col in submission_cols:
            cond = cond | pl.col(col).is_in(top_ids)
    else:
        rating = pl.concat(
            [df.select(pl.col(c).alias("mu")) for c in rating_cols]
        ).filter(pl.col("mu") > 0)
        if rating.is_empty():
            return df, 0.0
        cutoff = float(rating.quantile(rating_quantile).item())
        cond = pl.lit(False)
        for col in rating_cols:
            cond = cond | (pl.col(col) >= cutoff)
    df = df.filter(cond)
    if turn_min is not None or turn_max is not None:
        kept_match_ids = [
            str(row["match_id"])
            for row in df.select(["match_id", "replay_path"]).to_dicts()
            if _turn_count_in_range(
                base_dir / str(row["replay_path"]), turn_min, turn_max
            )
        ]
        df = df.filter(pl.col("match_id").is_in(kept_match_ids))
    return df, cutoff


def _turn_count_in_range(
    replay_path: Path,
    turn_min: int | None,
    turn_max: int | None,
) -> bool:
    if not replay_path.exists():
        return False
    data = _load_replay_json(replay_path)
    turns = len(data.get("steps", []))
    if turn_min is not None and turns < turn_min:
        return False
    if turn_max is not None and turns > turn_max:
        return False
    return True


def _index_arrow_schema() -> pa.Schema:
    """Side index schema, row-aligned with the mart parquet.

    Each row in train.parquet / val.parquet has one matching row here whose
    ``row_idx`` equals the position in the mart file. Captures upstream
    metadata so other analyses can re-filter without re-running preprocess.
    """
    return pa.schema(
        [
            pa.field("row_idx", pa.int64()),
            pa.field("match_id", pa.string()),
            pa.field("player_slot", pa.int32()),
            pa.field("source_slot", pa.int32()),
            pa.field("winner_slot", pa.int32()),
            pa.field("is_loser_view", pa.bool_()),
            pa.field("step_idx", pa.int32()),
            pa.field("turn_count", pa.int32()),
            pa.field("mode", pa.string()),
            pa.field("p0_submission_id", pa.string()),
            pa.field("p1_submission_id", pa.string()),
            pa.field("p0_rating_mu", pa.float32()),
            pa.field("p1_rating_mu", pa.float32()),
        ]
    )


def _arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("planet_feats", pa.list_(pa.float32())),
            pa.field("global_feats", pa.list_(pa.float32())),
            pa.field("planet_mask", pa.list_(pa.bool_())),
            pa.field("my_planet_mask", pa.list_(pa.bool_())),
            pa.field("target_mask", pa.list_(pa.bool_())),
            pa.field("template_ctx", pa.list_(pa.float32())),
            pa.field("candidate_feats", pa.list_(pa.float32())),
            pa.field("candidate_mask", pa.list_(pa.bool_())),
            pa.field("candidate_pid", pa.list_(pa.int32())),
            pa.field("cand_slot_per_src", pa.list_(pa.int32())),
            pa.field("ship_label_per_src", pa.list_(pa.int32())),
            pa.field("ships_bucket_per_src", pa.list_(pa.int32())),
            pa.field("from_multihot", pa.list_(pa.bool_())),
            pa.field("target_per_src", pa.list_(pa.int32())),
            pa.field("ships_per_src", pa.list_(pa.int32())),
            pa.field("is_noop", pa.bool_()),
        ]
    )


class StreamingParquetWriter:
    """Buffer rows and flush in chunks to keep RAM bounded.

    All rows for the parquet file go through this single instance; flushing
    is automatic when the buffer reaches ``flush_every`` rows. Call
    :meth:`close` after the last append to flush the tail and close the
    underlying ParquetWriter. ``rows_written`` reports the cumulative count.
    """

    def __init__(
        self,
        path: Path,
        flush_every: int = FLUSH_EVERY_FRAMES,
        *,
        schema: pa.Schema | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._flush_every = flush_every
        self._schema = schema if schema is not None else _arrow_schema()
        self._buf: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(str(path), self._schema, compression="zstd")
        self.rows_written = 0

    def append(self, row: dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf, schema=self._schema)
        self._writer.write_table(table)
        self.rows_written += len(self._buf)
        self._buf.clear()

    def close(self) -> int:
        self._flush()
        self._writer.close()
        return self.rows_written


def _num_player_slots(row: dict[str, Any], modes: list[str]) -> int:
    mode = str(row.get("mode", "1v1"))
    if mode == "ffa4":
        return 4
    return 2


def preprocess(cfg: dict[str, Any]) -> PreprocessReport:
    data_cfg = cfg["data"]
    modes = list(data_cfg.get("modes", ["1v1"]))
    rating_quantile = float(data_cfg.get("rating_quantile", 0.50))
    val_split = float(data_cfg.get("val_split", 0.10))
    max_episodes = data_cfg.get("max_episodes")
    top_submission_limit = data_cfg.get("top_submission_limit")
    turn_min = data_cfg.get("turn_min")
    turn_max = data_cfg.get("turn_max")
    loser_swap = bool(data_cfg.get("loser_swap", False))
    out_train = _abspath(data_cfg["out_train"])
    out_val = _abspath(data_cfg["out_val"])
    out_train_index = out_train.with_name(out_train.stem + "_index.parquet")
    out_val_index = out_val.with_name(out_val.stem + "_index.parquet")
    index_path = _abspath(data_cfg["kaggle_index_root"])
    base_dir = index_path.parent

    logger.info(
        "preprocess case10 start: planet_dim=%d global_dim=%d cand_K=%d cand_dim=%d "
        "modes=%s rating_q=%.2f top_submission_limit=%s turn_range=[%s,%s] "
        "val_split=%.2f max_episodes=%s",
        featurizer.PLANET_FEAT_DIM,
        featurizer.GLOBAL_FEAT_DIM,
        CAND_K,
        CAND_FEAT_DIM,
        modes,
        rating_quantile,
        top_submission_limit,
        turn_min,
        turn_max,
        val_split,
        max_episodes,
    )

    # The Kaggle episode index is a partitioned parquet dataset collected over
    # multiple scraper runs.  Newer partitions can contain extra metadata
    # columns (for example `agent_0_team_rank`) that older partitions do not
    # have.  Use scan_parquet with extra_columns="ignore" so the shared schema
    # remains readable on RunPod.
    index = pl.scan_parquet(index_path, extra_columns="ignore").collect()
    filtered, cutoff = _filter_index(
        index,
        modes,
        rating_quantile,
        base_dir=base_dir,
        top_submission_limit=(
            int(top_submission_limit) if top_submission_limit is not None else None
        ),
        turn_min=int(turn_min) if turn_min is not None else None,
        turn_max=int(turn_max) if turn_max is not None else None,
    )
    episodes_total = filtered.height
    if max_episodes is not None:
        filtered = filtered.head(int(max_episodes))

    rows = filtered.to_dicts()
    n_to_process = len(rows)
    logger.info(
        "preprocess case10 filter: cutoff=%.2f episodes_total=%d to_process=%d",
        cutoff,
        episodes_total,
        n_to_process,
    )

    train_writer = StreamingParquetWriter(out_train)
    val_writer = StreamingParquetWriter(out_val)
    train_index_writer = StreamingParquetWriter(
        out_train_index, schema=_index_arrow_schema()
    )
    val_index_writer = StreamingParquetWriter(
        out_val_index, schema=_index_arrow_schema()
    )

    started_at = time.monotonic()
    kept = 0
    skipped_no_slots = 0
    skipped_no_replay = 0
    skipped_no_frames = 0
    fired_total = 0
    outside_total = 0

    def _log_progress(processed: int) -> None:
        elapsed = time.monotonic() - started_at
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (n_to_process - processed) / rate if rate > 0 else float("inf")
        outside_pct = 100.0 * outside_total / max(1, fired_total)
        logger.info(
            "preprocess progress: %d/%d (%.1f%%) kept=%d "
            "skip(no_slots=%d, no_replay=%d, no_frames=%d) "
            "frames(train=%d val=%d) fired=%d outside=%d (%.1f%%) "
            "elapsed=%.1fs rate=%.2f ep/s eta=%.0fs",
            processed,
            n_to_process,
            100.0 * processed / max(1, n_to_process),
            kept,
            skipped_no_slots,
            skipped_no_replay,
            skipped_no_frames,
            train_writer.rows_written,
            val_writer.rows_written,
            fired_total,
            outside_total,
            outside_pct,
            elapsed,
            rate,
            remaining,
        )

    workers_env = os.environ.get("ORBIT_WARS_PREPROCESS_WORKERS")
    if workers_env is not None:
        max_workers = max(0, int(workers_env))
    else:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    # inflight cap = 4x workers — bounds memory of pending future results
    inflight_cap = max(4, max_workers * 4)
    logger.info(
        "preprocess case10 parallel: workers=%d inflight_cap=%d (override via "
        "ORBIT_WARS_PREPROCESS_WORKERS=0 for serial execution)",
        max_workers,
        inflight_cap,
    )

    rec_by_match: dict[str, dict[str, Any]] = {
        str(r["match_id"]): r for r in rows
    }

    def _consume_result(result: _EpisodeResult) -> None:
        nonlocal kept, fired_total, outside_total
        nonlocal skipped_no_slots, skipped_no_replay, skipped_no_frames
        if result.skip_reason == "no_slots":
            skipped_no_slots += 1
            return
        if result.skip_reason == "no_replay":
            skipped_no_replay += 1
            return
        if result.skip_reason == "no_frames":
            skipped_no_frames += 1
            fired_total += result.fired
            outside_total += result.outside
            return
        writer = val_writer if result.bucket == "val" else train_writer
        index_writer = val_index_writer if result.bucket == "val" else train_index_writer
        rec = rec_by_match.get(result.match_id, {})
        meta_static = _episode_meta_fields(rec)
        for frame, meta in zip(result.frames, result.metas, strict=True):
            row_idx = writer.rows_written + len(writer._buf)  # noqa: SLF001
            writer.append(frame)
            index_writer.append(
                {
                    "row_idx": int(row_idx),
                    "match_id": result.match_id,
                    "player_slot": int(meta["player_slot"]),
                    "source_slot": int(meta["source_slot"]),
                    "winner_slot": int(result.winner_slot),
                    "is_loser_view": bool(meta["is_loser_view"]),
                    "step_idx": int(meta["step_idx"]),
                    **meta_static,
                }
            )
        kept += 1
        fired_total += result.fired
        outside_total += result.outside

    try:
        if max_workers <= 1:
            # Serial path — same behavior as the original single-process loop.
            for processed, rec in enumerate(rows, start=1):
                try:
                    result = _process_episode(
                        rec,
                        base_dir,
                        val_split,
                        _num_player_slots(rec, modes),
                        loser_swap=loser_swap,
                    )
                    _consume_result(result)
                finally:
                    if processed % PROGRESS_LOG_EVERY == 0 or processed == n_to_process:
                        _log_progress(processed)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                in_iter = iter(enumerate(rows, start=1))
                inflight: dict[Any, int] = {}

                def _submit_next() -> bool:
                    try:
                        idx, rec = next(in_iter)
                    except StopIteration:
                        return False
                    fut = pool.submit(
                        _process_episode,
                        rec,
                        base_dir,
                        val_split,
                        _num_player_slots(rec, modes),
                        loser_swap=loser_swap,
                    )
                    inflight[fut] = idx
                    return True

                # prime the pool up to inflight_cap
                for _ in range(inflight_cap):
                    if not _submit_next():
                        break

                completed = 0
                while inflight:
                    for fut in as_completed(list(inflight)):
                        idx = inflight.pop(fut)
                        try:
                            result = fut.result()
                            _consume_result(result)
                        except Exception:
                            logger.exception(
                                "preprocess worker failed on episode idx=%d", idx
                            )
                            skipped_no_frames += 1
                        completed += 1
                        if (
                            completed % PROGRESS_LOG_EVERY == 0
                            or completed == n_to_process
                        ):
                            _log_progress(completed)
                        # Refill so total inflight stays at inflight_cap
                        _submit_next()
                        # Re-poll as_completed after refill so that newly
                        # submitted futures also enter the wait set; break the
                        # inner loop and re-check the outer `while inflight`.
                        break
    finally:
        logger.info(
            "preprocess case10 closing parquet writers: "
            "train_flushed_so_far=%d val_flushed_so_far=%d",
            train_writer.rows_written,
            val_writer.rows_written,
        )
        close_started = time.monotonic()
        n_train = train_writer.close()
        n_val = val_writer.close()
        n_train_idx = train_index_writer.close()
        n_val_idx = val_index_writer.close()
        if n_train_idx != n_train or n_val_idx != n_val:
            logger.warning(
                "preprocess case10 index/mart row mismatch: "
                "train mart=%d index=%d / val mart=%d index=%d",
                n_train,
                n_train_idx,
                n_val,
                n_val_idx,
            )
        logger.info(
            "preprocess case10 parquet finalized: train=%d val=%d close_elapsed=%.1fs "
            "out_train=%s out_val=%s out_train_index=%s out_val_index=%s "
            "loser_swap=%s",
            n_train,
            n_val,
            time.monotonic() - close_started,
            out_train,
            out_val,
            out_train_index,
            out_val_index,
            loser_swap,
        )

    report = PreprocessReport(
        rating_cutoff=cutoff,
        episodes_total=episodes_total,
        episodes_kept=kept,
        train_frames=n_train,
        val_frames=n_val,
        out_train=out_train,
        out_val=out_val,
        label_outside_candidates=outside_total,
        fired_total=fired_total,
    )
    pct = 100.0 * outside_total / max(1, fired_total)
    total_elapsed = time.monotonic() - started_at
    logger.info(
        "preprocess case10 done: cutoff=%.2f kept=%d/%d "
        "skip(no_slots=%d, no_replay=%d, no_frames=%d) "
        "frames train=%d val=%d label_outside_K=%d / fired=%d (%.1f%%) "
        "total_elapsed=%.1fs",
        report.rating_cutoff,
        report.episodes_kept,
        report.episodes_total,
        skipped_no_slots,
        skipped_no_replay,
        skipped_no_frames,
        report.train_frames,
        report.val_frames,
        outside_total,
        fired_total,
        pct,
        total_elapsed,
    )
    return report


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case10/configs/il_case10_candidate.yaml"),
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
        f"train_frames={report.train_frames} val_frames={report.val_frames} "
        f"label_outside={report.label_outside_candidates}/{report.fired_total}"
    )


if __name__ == "__main__":
    app()
