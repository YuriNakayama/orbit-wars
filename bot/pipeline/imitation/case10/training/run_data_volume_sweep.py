"""Driver for case10 data_volume_sweep (iter1 + iter2).

Runs N training points against the **shared base mart**
(`data/mart/imitation/case10/base/{train,val}.parquet` + `*_index.parquet`).

Per point we (a) write a sweep config that overrides
``data.train_top_submission_limit`` and ``data.use_loser_swap`` while pointing
``data.out_train`` / ``data.out_val`` at the base mart, (b) run the train
module which in turn applies the in-memory index filter via
``index_filter.compute_keep_row_indices``, then (c) eval the produced weights
vs ``baseline_v1``.

Output: ``data/output/experiment/case10_data_volume_sweep/summary.json`` keyed
by tag, containing wall-clock + best val metrics + win-rate.

Run from ``bot/``::

    # iter1 (5 sweep points: top20/top40/top80/top160/topall)
    uv run python -m pipeline.imitation.case10.training.run_data_volume_sweep \
        --iter 1

    # iter2 (2 swap on/off points; reuses iter1's top80 result if present)
    uv run python -m pipeline.imitation.case10.training.run_data_volume_sweep \
        --iter 2
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT_FROM_BOT = Path("..").resolve()
BASE_TRAIN = "data/mart/imitation/case10/base/train.parquet"
BASE_VAL = "data/mart/imitation/case10/base/val.parquet"
WEIGHTS_DIR = Path("pipeline/imitation/case10/policy/data_volume_sweep")
SUMMARY_PATH = Path("data/output/experiment/case10_data_volume_sweep/summary.json")
TEMPLATE_CFG = Path("pipeline/imitation/case10/configs/il_case10_template.yaml")
DEFAULT_HEAD_MODE = "template_ships"

# (tag, train_top_submission_limit, use_loser_swap)
# iter1 sweep was originally top20/40/80/160/all (5 points). On 2026-05-12
# the 13 GB base mart couldn't be DVC-pushed reliably (signature expiry and
# SSL EOF mid-upload) and topall/top160 exceeded local RAM at load time, so
# the scope was reduced to top20/40/80; top80 doubles as the H5 reference
# point in the adoption rule (see hypotheses.md).
ITER_POINTS: dict[int, list[tuple[str, int | None, bool]]] = {
    1: [
        ("top20", 20, True),
        ("top40", 40, True),
        ("top80", 80, True),
    ],
    2: [
        ("top80_swap_on", 80, True),
        ("top80_swap_off", 80, False),
    ],
}


def _run(cmd: list[str]) -> tuple[float, int, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return elapsed, proc.returncode, proc.stdout + proc.stderr


def _materialize_config(
    base_cfg_path: Path, tag: str, top_K: int | None, use_loser_swap: bool
) -> Path:
    """Write a temp YAML that overrides data.* fields for this sweep point."""
    cfg = yaml.safe_load(base_cfg_path.read_text())
    data = dict(cfg.get("data", {}))
    data["out_train"] = BASE_TRAIN
    data["out_val"] = BASE_VAL
    data["train_top_submission_limit"] = top_K
    data["use_loser_swap"] = use_loser_swap
    cfg["data"] = data
    train = dict(cfg.get("train", {}))
    train["weights_out"] = str(WEIGHTS_DIR / f"weights_{tag}.pt")
    cfg["train"] = train
    out_path = Path(tempfile.gettempdir()) / f"case10_sweep_{tag}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def _train(cfg_path: Path) -> tuple[float, int, str]:
    return _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pipeline.imitation.case10.training.train",
            "--config",
            str(cfg_path),
        ]
    )


def _eval(
    weights: Path, tag: str, head_mode: str, episodes_per_side: int
) -> tuple[float, int, str]:
    return _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pipeline.imitation.case10.evaluation.eval_data_volume_sweep",
            "--weights",
            str(weights),
            "--tag",
            tag,
            "--head-mode",
            head_mode,
            "--episodes-per-side",
            str(episodes_per_side),
        ]
    )


def _read_eval_summary(tag: str) -> dict:
    p = SUMMARY_PATH.parent / f"eval_{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _read_train_metrics(weights: Path) -> dict:
    """Look for a train_metrics.json next to the weights file (run_dir)."""
    candidates = [
        weights.with_suffix(".json"),
        weights.parent / "train_metrics.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except json.JSONDecodeError:
                continue
    return {}


def _run_one_point(
    tag: str,
    top_K: int | None,
    use_loser_swap: bool,
    *,
    base_cfg_path: Path,
    head_mode: str,
    episodes_per_side: int,
    skip_train: bool,
    skip_eval: bool,
) -> dict:
    cfg_path = _materialize_config(base_cfg_path, tag, top_K, use_loser_swap)
    weights = WEIGHTS_DIR / f"weights_{tag}.pt"
    record: dict = {
        "tag": tag,
        "top_submission_limit": top_K,
        "use_loser_swap": use_loser_swap,
        "config": str(cfg_path),
        "weights": str(weights),
    }

    if not skip_train:
        logger.info("[%s] train start", tag)
        elapsed, rc, _ = _train(cfg_path)
        record["train_seconds"] = round(elapsed, 2)
        record["train_rc"] = rc
        if rc != 0:
            record["status"] = "train_failed"
            return record
        record.update({"train_metrics": _read_train_metrics(weights)})

    if skip_eval or not weights.exists():
        record["status"] = "ok_no_eval"
        return record

    logger.info("[%s] eval start (vs baseline_v1, %d×P1/P2)", tag, episodes_per_side)
    elapsed, rc, _ = _eval(weights, tag, head_mode, episodes_per_side)
    record["eval_seconds"] = round(elapsed, 2)
    record["eval_rc"] = rc
    if rc != 0:
        record["status"] = "eval_failed"
        return record
    record["eval_summary"] = _read_eval_summary(tag)
    record["status"] = "ok"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iter",
        dest="iter_n",
        type=int,
        default=1,
        choices=sorted(ITER_POINTS),
        help="iter1=top_K sweep / iter2=swap on-off",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=TEMPLATE_CFG,
        help="case10 base config (defines model + train hparams)",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default=DEFAULT_HEAD_MODE,
        help="match the head_mode of the base config",
    )
    parser.add_argument("--episodes-per-side", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SUMMARY_PATH.exists():
        try:
            results: list[dict] = json.loads(SUMMARY_PATH.read_text())
        except json.JSONDecodeError:
            results = []
    else:
        results = []
    by_tag = {r["tag"]: i for i, r in enumerate(results)}

    points = ITER_POINTS[args.iter_n]
    for tag, top_K, use_loser_swap in points:
        rec = _run_one_point(
            tag,
            top_K,
            use_loser_swap,
            base_cfg_path=args.base_config,
            head_mode=args.head_mode,
            episodes_per_side=args.episodes_per_side,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
        )
        rec["iter"] = args.iter_n
        if tag in by_tag:
            results[by_tag[tag]] = rec
        else:
            by_tag[tag] = len(results)
            results.append(rec)
        SUMMARY_PATH.write_text(json.dumps(results, indent=2))
        logger.info("[%s] done: %s", tag, rec.get("status"))

    logger.info("summary written to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
