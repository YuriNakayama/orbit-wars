"""Replay → parquet preprocess for imitation/case11 (per_planet head only).

Per-frame supervision schema (3-layer mask design):

  - planet_feats : (MAX_PLANETS * PLANET_FEAT_DIM,) float32
  - global_feats : (GLOBAL_FEAT_DIM,)              float32
  - planet_mask  : (MAX_PLANETS,)                  bool  — layer1: existence
  - target_mask  : (MAX_PLANETS,)                  bool  — layer1: physical-rule
  - candidate_feats : (MAX_PLANETS * CAND_K * CAND_FEAT_DIM,) float32
  - candidate_mask  : (MAX_PLANETS * CAND_K,)      bool
  - candidate_pid   : (MAX_PLANETS * CAND_K,)      int32
  - effective_source_mask : (MAX_PLANETS,)         bool  — layer2: my_planet & ships>0
  - should_learn_ship     : (MAX_PLANETS,)         bool  — layer3: fire-only ship_head
  - target_pid_per_src    : (MAX_PLANETS,)         int32 (0..P-1=fire target slot,
                                                         P = no-op sentinel)
  - ship_pred_label       : (MAX_PLANETS,)         float32 (log1p(ships) for
                                                            fired sources, -1.0
                                                            otherwise)
  - is_noop : bool

Mask layering (Lux3-derived):
  layer1 (invalid-action): target_mask -1e9 on softmax (decoder side)
  layer2 (existence/ownership): effective_source_mask drives the target_head
         source axis. encoder still sees all planets (enemy/neutral as
         observation context).
  layer3 (conditional head): ship_head loss only on should_learn_ship rows.
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

from pipeline.imitation.case11.policy import featurizer
from pipeline.imitation.case11.policy.featurizer import (
    MAX_PLANETS,
    HistoryState,
)
from pipeline.imitation.case11.policy.geometry import Planet, aim_with_prediction

logger = logging.getLogger(__name__)

ANGLE_TOLERANCE = 0.20  # radians (~11.5deg) — angle reverse-resolve match window
PROGRESS_LOG_EVERY = 100  # log every N episodes processed (kept or skipped)
FLUSH_EVERY_FRAMES = 5000  # flush parquet rows when buffer reaches this size


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


def _select_winner_slots(
    rec: dict[str, Any],
    num_slots: int,
    *,
    winners_only: bool,
    top_team_rank: int,
) -> list[int]:
    """Pick which player slots to emit frames for, after winner / rank filters.

    - winners_only=True: keep only the winning slot (per `winner` field).
    - top_team_rank > 0: drop the slot if its `agent_{slot}_team_rank` is
      either 0 (unknown) or above the threshold. Top-80 → top_team_rank=80.
    """
    if rec.get("draw"):
        return []
    if winners_only:
        winner = int(rec.get("winner", -1))
        if winner < 0 or winner >= num_slots:
            return []
        slots = [winner]
    else:
        slots = list(range(num_slots))
    if top_team_rank > 0:
        kept: list[int] = []
        for slot in slots:
            rank = int(rec.get(f"agent_{slot}_team_rank", 0) or 0)
            if rank <= 0 or rank > top_team_rank:
                continue
            kept.append(slot)
        slots = kept
    return slots


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
    effective_source_mask = batch.effective_source_mask[0].numpy().astype(np.bool_)
    target_mask = batch.target_mask[0].numpy().astype(np.bool_)
    cand_feats = batch.candidate_feats[0].numpy().astype(np.float32)
    cand_mask = batch.candidate_mask[0].numpy().astype(np.bool_)
    cand_pid = batch.candidate_pid[0].numpy().astype(np.int32)
    template_ctx = batch.template_ctx[0].numpy().astype(np.float32)
    player = int(obs.get("player", 0) or 0)

    pid_to_slot = {pid: i for i, pid in enumerate(snap.planet_ids)}
    planets = [_build_planet(row) for row in raw_planets]
    by_id = {p.id: p for p in planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    # my_planet_mask は label 生成の素材としてのみ使う。BatchFeatures には
    # effective_source_mask (my_planet & ships>0) しか乗らない。preprocess は
    # 「自軍所有」自体を素材として参照する必要があるので、ships=0 source の
    # default no-op label を埋めるためにここで再計算する。
    my_planet_mask = np.zeros(MAX_PLANETS, dtype=np.bool_)
    for slot, pid in enumerate(snap.planet_ids):
        if pid < 0 or slot >= MAX_PLANETS:
            continue
        planet = by_id.get(int(pid))
        if planet is None:
            continue
        if int(planet.owner) == player:
            my_planet_mask[slot] = True

    # per_planet head 用ラベル: target slot 0..MAX_PLANETS-1 = fire to that slot,
    # MAX_PLANETS = no-op sentinel.
    target_pid_per_src = np.full(MAX_PLANETS, MAX_PLANETS, dtype=np.int32)
    # ship_pred_label: log1p(ships) for fired sources, -1.0 otherwise.
    ship_pred_label = np.full(MAX_PLANETS, -1.0, dtype=np.float32)

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
        # per_planet ships regression label (log1p space).
        ship_pred_label[slot] = float(math.log1p(max(0, ships)))
        if target_pid is None:
            outside += 1
            continue
        # per_planet target slot label (planet index in pid_to_slot).
        tgt_slot = pid_to_slot.get(int(target_pid))
        if tgt_slot is not None and 0 <= tgt_slot < MAX_PLANETS:
            target_pid_per_src[slot] = int(tgt_slot)
        else:
            outside += 1

    is_noop = bool(np.all(target_pid_per_src[my_planet_mask] == MAX_PLANETS))

    # effective_source_mask は featurizer 由来 (BatchFeatures から既に抽出)。
    # should_learn_ship (layer3) は teacher action が "fire" の source のみ
    # ship_head に学習させるためのマスク。
    no_op_idx = MAX_PLANETS
    should_learn_ship = effective_source_mask & (target_pid_per_src != no_op_idx)

    row = {
        "planet_feats": planet_feats.reshape(-1).tolist(),
        "global_feats": global_feats.tolist(),
        "planet_mask": planet_mask.tolist(),
        "target_mask": target_mask.tolist(),
        "template_ctx": template_ctx.reshape(-1).tolist(),
        "candidate_feats": cand_feats.reshape(-1).tolist(),
        "candidate_mask": cand_mask.reshape(-1).tolist(),
        "candidate_pid": cand_pid.reshape(-1).tolist(),
        "effective_source_mask": effective_source_mask.tolist(),
        "should_learn_ship": should_learn_ship.tolist(),
        "target_pid_per_src": target_pid_per_src.tolist(),
        "ship_pred_label": ship_pred_label.tolist(),
        "is_noop": is_noop,
    }
    return row, fired_total, outside


def _iter_episode_frames(
    replay_path: Path,
    player_slots: list[int],
) -> tuple[list[dict[str, Any]], int, int]:
    with gzip.open(replay_path, "rt") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    out: list[dict[str, Any]] = []
    histories: dict[int, HistoryState] = {slot: HistoryState() for slot in player_slots}
    fired_total = 0
    outside_total = 0

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

            history = histories[slot]
            built = _build_frame(obs, action_list, planets, history)
            if built is not None:
                row, fired, outside = built
                out.append(row)
                fired_total += fired
                outside_total += outside
            featurizer.update_history(history, obs, action_list)
    return out, fired_total, outside_total


@dataclass(frozen=True)
class _EpisodeResult:
    bucket: str  # "train" | "val"
    frames: list[dict[str, Any]]
    fired: int
    outside: int
    skip_reason: str | None  # None | "no_slots" | "no_replay" | "no_frames"


def _process_episode(
    rec: dict[str, Any],
    base_dir: Path,
    val_split: float,
    num_player_slots: int,
    *,
    winners_only: bool = False,
    top_team_rank: int = 0,
) -> _EpisodeResult:
    """Worker: process one episode record into frames.

    Pure function (importable / picklable). Returns a single result.
    """
    if num_player_slots == 0 or rec.get("draw"):
        return _EpisodeResult(
            bucket="", frames=[], fired=0, outside=0, skip_reason="no_slots"
        )
    slots = _select_winner_slots(
        rec,
        num_player_slots,
        winners_only=winners_only,
        top_team_rank=top_team_rank,
    )
    if not slots:
        return _EpisodeResult(
            bucket="", frames=[], fired=0, outside=0, skip_reason="no_slots"
        )
    rp = rec.get("replay_path") or ""
    replay_path = (base_dir / rp).resolve() if rp else None
    if replay_path is None or not replay_path.exists():
        return _EpisodeResult(
            bucket="", frames=[], fired=0, outside=0, skip_reason="no_replay"
        )
    frames, fired, outside = _iter_episode_frames(replay_path, slots)
    if not frames:
        return _EpisodeResult(
            bucket="", frames=[], fired=fired, outside=outside, skip_reason="no_frames"
        )
    bucket = _split_episode(str(rec["match_id"]), val_split)
    return _EpisodeResult(
        bucket=bucket, frames=frames, fired=fired, outside=outside, skip_reason=None
    )


def _filter_index(
    index: pl.DataFrame,
    modes: list[str],
    rating_quantile: float,
    recent_days: int | None = None,
    top_winner_rank: int | None = None,
) -> tuple[pl.DataFrame, float]:
    df = index.filter(pl.col("mode").is_in(modes))
    df = df.filter(~pl.col("draw"))
    if recent_days is not None and recent_days > 0:
        start_d = df["started_at"].str.slice(0, 10).str.to_date(format="%Y-%m-%d")
        max_raw = start_d.max()
        if not isinstance(max_raw, datetime.date):
            raise RuntimeError(f"started_at max is not a date: {max_raw!r}")
        cutoff_d = max_raw - datetime.timedelta(days=int(recent_days))
        df = (
            df.with_columns(start_d.alias("_start_d"))
            .filter(pl.col("_start_d") >= cutoff_d)
            .drop("_start_d")
        )
    if top_winner_rank is not None and top_winner_rank > 0:
        winner_rank = (
            pl.when(pl.col("winner") == 0)
            .then(pl.col("agent_0_team_rank"))
            .otherwise(pl.col("agent_1_team_rank"))
        )
        df = df.filter((winner_rank > 0) & (winner_rank <= int(top_winner_rank)))
    rating_cols = [c for c in df.columns if c.endswith("_rating_mu")]
    rating = pl.concat([df.select(pl.col(c).alias("mu")) for c in rating_cols]).filter(
        pl.col("mu") > 0
    )
    if rating.is_empty():
        return df, 0.0
    cutoff = float(rating.quantile(rating_quantile).item())
    cond = pl.lit(False)
    for col in rating_cols:
        cond = cond | (pl.col(col) >= cutoff)
    df = df.filter(cond)
    return df, cutoff


def _arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("planet_feats", pa.list_(pa.float32())),
            pa.field("global_feats", pa.list_(pa.float32())),
            pa.field("planet_mask", pa.list_(pa.bool_())),
            pa.field("target_mask", pa.list_(pa.bool_())),
            pa.field("template_ctx", pa.list_(pa.float32())),
            pa.field("candidate_feats", pa.list_(pa.float32())),
            pa.field("candidate_mask", pa.list_(pa.bool_())),
            pa.field("candidate_pid", pa.list_(pa.int32())),
            pa.field("effective_source_mask", pa.list_(pa.bool_())),
            pa.field("should_learn_ship", pa.list_(pa.bool_())),
            pa.field("target_pid_per_src", pa.list_(pa.int32())),
            pa.field("ship_pred_label", pa.list_(pa.float32())),
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

    def __init__(self, path: Path, flush_every: int = FLUSH_EVERY_FRAMES) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._flush_every = flush_every
        self._buf: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(str(path), _arrow_schema(), compression="zstd")
        self.rows_written = 0

    def append(self, row: dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf, schema=_arrow_schema())
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
    winners_only = bool(data_cfg.get("winners_only", False))
    top_team_rank = int(data_cfg.get("top_team_rank", 0) or 0)
    recent_days_cfg = data_cfg.get("recent_days")
    recent_days = int(recent_days_cfg) if recent_days_cfg else None
    top_winner_rank_cfg = data_cfg.get("top_winner_rank")
    top_winner_rank = int(top_winner_rank_cfg) if top_winner_rank_cfg else None
    out_train = _abspath(data_cfg["out_train"])
    out_val = _abspath(data_cfg["out_val"])
    index_path = _abspath(data_cfg["kaggle_index_root"])
    base_dir = index_path.parent

    logger.info(
        "preprocess case11 start: planet_dim=%d global_dim=%d "
        "modes=%s rating_q=%.2f val_split=%.2f max_episodes=%s "
        "winners_only=%s top_team_rank=%d recent_days=%s top_winner_rank=%s",
        featurizer.PLANET_FEAT_DIM,
        featurizer.GLOBAL_FEAT_DIM,
        modes,
        rating_quantile,
        val_split,
        max_episodes,
        winners_only,
        top_team_rank,
        recent_days,
        top_winner_rank,
    )

    # Hive-partitioned index parquet has heterogeneous schemas across files —
    # the older files predate `agent_*_team_rank`. `polars.scan_parquet` locks
    # the union schema to the first file it sees, which drops `team_rank`
    # entirely. Use `pyarrow.dataset` instead: it unifies schemas across all
    # discovered files, filling missing columns with NULL.
    import pyarrow.dataset as pa_ds

    parquet_files: list[str] = []
    for dirpath, _dirs, files in os.walk(str(index_path)):
        for fname in files:
            if fname.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, fname))
    if not parquet_files:
        raise RuntimeError(f"no parquet files found under {index_path}")
    arrow_table = pa_ds.dataset(parquet_files, format="parquet").to_table()
    index = pl.from_arrow(arrow_table)
    assert isinstance(index, pl.DataFrame)
    filtered, cutoff = _filter_index(
        index,
        modes,
        rating_quantile,
        recent_days=recent_days,
        top_winner_rank=top_winner_rank,
    )
    episodes_total = filtered.height
    if max_episodes is not None:
        filtered = filtered.head(int(max_episodes))

    rows = filtered.to_dicts()
    n_to_process = len(rows)
    logger.info(
        "preprocess case8 filter: cutoff=%.2f episodes_total=%d to_process=%d",
        cutoff,
        episodes_total,
        n_to_process,
    )

    train_writer = StreamingParquetWriter(out_train)
    val_writer = StreamingParquetWriter(out_val)

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
        "preprocess case8 parallel: workers=%d inflight_cap=%d (override via "
        "ORBIT_WARS_PREPROCESS_WORKERS=0 for serial execution)",
        max_workers,
        inflight_cap,
    )

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
        for frame in result.frames:
            writer.append(frame)
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
                        winners_only=winners_only,
                        top_team_rank=top_team_rank,
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
                        winners_only=winners_only,
                        top_team_rank=top_team_rank,
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
            "preprocess case8 closing parquet writers: "
            "train_flushed_so_far=%d val_flushed_so_far=%d",
            train_writer.rows_written,
            val_writer.rows_written,
        )
        close_started = time.monotonic()
        n_train = train_writer.close()
        n_val = val_writer.close()
        logger.info(
            "preprocess case8 parquet finalized: train=%d val=%d close_elapsed=%.1fs "
            "out_train=%s out_val=%s",
            n_train,
            n_val,
            time.monotonic() - close_started,
            out_train,
            out_val,
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
        "preprocess case8 done: cutoff=%.2f kept=%d/%d "
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
        Path("pipeline/imitation/case8/configs/il_case8.yaml"),
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
