"""Replay → parquet preprocess for imitation/case3 IL baseline.

One row per (obs, player) frame. Multi-hot from-set + per-source target/ships
labels (slot -1 for "not selected" sources). Both winner and loser sides are
included.
"""

from __future__ import annotations

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

from dataset.storage.loader import load_replay_payload_from_uri
from pipeline.imitation.case3.policy import featurizer_phase2
from pipeline.imitation.case3.policy.featurizer_phase2 import (
    MAX_PLANETS,
    HistoryState,
)
from pipeline.imitation.case3.policy.geometry import Planet, aim_with_prediction
from pipeline.imitation.case3.policy.templates import (
    T_NO_OP,
    classify_actual_target,
)

FEATURIZERS = {
    "phase2": featurizer_phase2,
}
PHASE2 = "phase2"

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """`bot/` の親ディレクトリ (リポジトリ root) を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bot").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _abspath(rel: str | Path) -> Path:
    """相対パスを repo root 起点の絶対パスに解決する.

    Vast.ai 上では `uv run --directory bot ...` で CWD が bot/ になるため、
    config の `data/...` 相対パスをそのまま `Path(...)` で開くと
    `bot/data/...` を見に行ってしまい FileNotFoundError になる。常に
    repo root 起点で解決して CWD 非依存にする (case1 preprocess と同パターン)。
    """
    p = Path(rel)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


UNUSED_LABEL = -1
SHIPS_BUCKETS = 4  # 0:25%, 1:50%, 2:75%, 3:100%
ANGLE_TOLERANCE = 0.20  # radians (~11.5deg) — angle reverse-resolve match window


@dataclass(frozen=True)
class PreprocessReport:
    rating_cutoff: float
    episodes_total: int
    episodes_kept: int
    train_frames: int
    val_frames: int
    out_train: Path
    out_val: Path


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
    """Reverse-resolve fired (angle, ships) into the most likely target planet id.

    Walks every non-source planet, calls aim_with_prediction (the same routine the
    pro player uses) and picks the candidate whose predicted angle is closest to
    the actually-fired angle (within ANGLE_TOLERANCE).
    """
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


def _ships_bucket(ships: int, src_ships: int) -> int:
    if src_ships <= 0:
        return 0
    ratio = max(0.0, min(1.0, ships / max(1, src_ships)))
    # Map (0, 1] to bucket [0, SHIPS_BUCKETS-1]:
    #  ratio in (0,    0.375] → 0 (~25%)
    #  ratio in (0.375,0.625] → 1 (~50%)
    #  ratio in (0.625,0.875] → 2 (~75%)
    #  ratio in (0.875,1.0  ] → 3 (~100%)
    bucket = int(round(ratio * (SHIPS_BUCKETS - 1) - 0.001))
    return max(0, min(SHIPS_BUCKETS - 1, bucket))


def _split_episode(match_id: str, val_split: float) -> str:
    h = hashlib.sha1(match_id.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], "big") / 2**32
    return "val" if val < val_split else "train"


def _player_slots(row: dict[str, Any], num_slots: int) -> list[int]:
    if row.get("draw"):
        return []
    return list(range(num_slots))


def _build_frame(
    obs: dict[str, Any],
    action_list: list[list[Any]],
    raw_planets: list[list[Any]],
    featurize_fn: Any,
) -> dict[str, Any] | None:
    """Build one parquet row representing the frame's full action set."""
    batch, snap = featurize_fn(obs)
    planet_feats = batch.planet_feats[0].numpy().astype(np.float32)
    global_feats = batch.global_feats[0].numpy().astype(np.float32)
    planet_mask = batch.planet_mask[0].numpy().astype(np.bool_)
    my_planet_mask = batch.my_planet_mask[0].numpy().astype(np.bool_)
    target_mask = batch.target_mask[0].numpy().astype(np.bool_)
    template_ctx = batch.template_ctx[0].numpy().astype(np.float32)

    pid_to_slot = {pid: i for i, pid in enumerate(snap.planet_ids)}
    planets = [_build_planet(row) for row in raw_planets]
    by_id = {p.id: p for p in planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    from_multihot = np.zeros(MAX_PLANETS, dtype=np.bool_)
    target_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)
    ships_per_src = np.full(MAX_PLANETS, UNUSED_LABEL, dtype=np.int32)

    player = int(obs.get("player", 0) or 0)
    for act in action_list:
        from_pid = int(act[0])
        angle = float(act[1])
        ships = int(act[2])
        slot = pid_to_slot.get(from_pid)
        if slot is None or not my_planet_mask[slot]:
            continue
        src = by_id.get(from_pid)
        if src is None:
            continue
        target_pid = _resolve_action_target(
            src, angle, ships, planets, initial_by_id, ang_vel, comets, comet_ids
        )
        if target_pid is None or target_pid not in pid_to_slot:
            template_id = T_NO_OP
        else:
            src_row = raw_planets[slot]
            tgt_row = raw_planets[pid_to_slot[target_pid]]
            template_id = classify_actual_target(src_row, tgt_row, raw_planets, player)
        from_multihot[slot] = True
        target_per_src[slot] = int(template_id)
        ships_per_src[slot] = int(_ships_bucket(ships, src.ships))

    is_noop = not bool(from_multihot.any())

    return {
        "planet_feats": planet_feats.reshape(-1).tolist(),
        "global_feats": global_feats.tolist(),
        "planet_mask": planet_mask.tolist(),
        "my_planet_mask": my_planet_mask.tolist(),
        "target_mask": target_mask.tolist(),
        "template_ctx": template_ctx.reshape(-1).tolist(),
        "from_multihot": from_multihot.tolist(),
        "target_per_src": target_per_src.tolist(),
        "ships_per_src": ships_per_src.tolist(),
        "is_noop": bool(is_noop),
    }


def _iter_episode_frames(
    replay_uri: str,
    player_slots: list[int],
    featurizer_mod: Any,
    featurizer_name: str,
) -> list[dict[str, Any]]:
    data = load_replay_payload_from_uri(replay_uri)
    steps = data.get("steps", [])
    out: list[dict[str, Any]] = []
    # Per-slot history for phase2: mutated in-place across an episode.
    histories: dict[int, Any] = {}
    if featurizer_name == PHASE2:
        for slot in player_slots:
            histories[slot] = HistoryState()

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
            # Kaggle replay の loser 側 obs は `step` が None なので注入する
            if obs.get("step") is None:
                obs = {**obs, "step": step_idx}
            if obs.get("player") is None:
                obs = {**obs, "player": slot}

            if featurizer_name == PHASE2:
                history = histories[slot]
                featurize_fn = lambda o, h=history: featurizer_mod.featurize(o, h)
            else:
                featurize_fn = featurizer_mod.featurize

            frame = _build_frame(obs, action_list, planets, featurize_fn)
            if frame is not None:
                out.append(frame)

            if featurizer_name == PHASE2:
                featurizer_mod.update_history(histories[slot], obs, action_list)
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

    # require AT LEAST ONE seat to clear cutoff (loser side included).
    cond = pl.lit(False)
    for col in rating_cols:
        cond = cond | (pl.col(col) >= cutoff)
    df = df.filter(cond)
    return df, cutoff


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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
        "template_ctx": pl.List(pl.Float32),
        "from_multihot": pl.List(pl.Boolean),
        "target_per_src": pl.List(pl.Int32),
        "ships_per_src": pl.List(pl.Int32),
        "is_noop": pl.Boolean(),
    }


def _num_player_slots(row: dict[str, Any], modes: list[str]) -> int:
    mode = str(row.get("mode", "1v1"))
    if mode == "ffa4":
        return 4
    return 2


def _duplicate_minority_frames(
    train_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Row-level augmentation: frames that fire a minority target template get
    duplicated `multiplier` times so their gradient contribution grows per
    epoch. Unlike WeightedRandomSampler (which picks the same row repeatedly),
    this increases the true count of minority frames in the parquet so shuffled
    training sees fresh pairings against non-duplicated rows each epoch.

    `cfg["templates"]`: list of template ids to treat as minority.
    `cfg["multiplier"]`: positive int. `2` = frame appears twice total.
    """
    templates = set(int(t) for t in cfg.get("templates", []))
    multiplier = int(cfg.get("multiplier", 2))
    if not templates or multiplier <= 1:
        return train_rows
    extra_copies = multiplier - 1
    out: list[dict[str, Any]] = []
    n_dupe = 0
    for row in train_rows:
        out.append(row)
        fired = [int(t) for t in row.get("target_per_src", []) if int(t) != -1]
        if any(t in templates for t in fired):
            out.extend(row for _ in range(extra_copies))
            n_dupe += extra_copies
    logger.info(
        "duplicate_minority: original=%d, duplicated_rows=%d, total=%d",
        len(train_rows),
        n_dupe,
        len(out),
    )
    return out


def preprocess(cfg: dict[str, Any]) -> PreprocessReport:
    data_cfg = cfg["data"]
    modes = list(data_cfg.get("modes", ["1v1"]))
    rating_quantile = float(data_cfg.get("rating_quantile", 0.75))
    val_split = float(data_cfg.get("val_split", 0.10))
    max_episodes = data_cfg.get("max_episodes")
    out_train = _abspath(data_cfg["out_train"])
    out_val = _abspath(data_cfg["out_val"])
    index_path = _abspath(data_cfg["kaggle_index_root"])
    featurizer_name = str(data_cfg.get("featurizer", "baseline"))
    if featurizer_name not in FEATURIZERS:
        raise ValueError(
            f"unknown featurizer={featurizer_name!r}, expected one of {list(FEATURIZERS)}"
        )
    featurizer_mod = FEATURIZERS[featurizer_name]
    logger.info(
        "preprocess featurizer=%s planet_dim=%d global_dim=%d",
        featurizer_name,
        featurizer_mod.PLANET_FEAT_DIM,
        featurizer_mod.GLOBAL_FEAT_DIM,
    )

    index = pl.read_parquet(index_path)
    filtered, cutoff = _filter_index(index, modes, rating_quantile)
    episodes_total = filtered.height
    if max_episodes is not None:
        filtered = filtered.head(int(max_episodes))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rows = filtered.to_dicts()
    expected_dim = MAX_PLANETS * featurizer_mod.PLANET_FEAT_DIM
    assert expected_dim > 0 and featurizer_mod.GLOBAL_FEAT_DIM > 0  # sanity

    kept = 0
    for rec in rows:
        slots = _player_slots(rec, _num_player_slots(rec, modes))
        if not slots:
            continue
        uri = rec.get("replay_uri") or ""
        if not uri:
            continue
        try:
            frames = _iter_episode_frames(uri, slots, featurizer_mod, featurizer_name)
        except (OSError, ValueError) as exc:
            logger.warning("skip replay %s: %s", uri, exc)
            continue
        if not frames:
            continue
        bucket = _split_episode(str(rec["match_id"]), val_split)
        target = val_rows if bucket == "val" else train_rows
        target.extend(frames)
        kept += 1

    duplicate_cfg = data_cfg.get("duplicate_minority", {}) or {}
    if duplicate_cfg.get("enabled", False):
        train_rows = _duplicate_minority_frames(train_rows, duplicate_cfg)

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
        Path("pipeline/imitation/case3/configs/il_baseline.yaml"),
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
