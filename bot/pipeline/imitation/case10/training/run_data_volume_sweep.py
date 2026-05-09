"""Driver for case10 data-volume sweep.

Iterates over the 5 sweep configs under
`pipeline/imitation/case10/configs/data_volume_sweep/`, runs preprocess + train
for each, and records wall-clock + best val metrics into a JSON summary at
`data/output/experiment/case10_data_volume_sweep/summary.json`.

Run from `bot/`:

    uv run python -m pipeline.imitation.case10.training.run_data_volume_sweep \
        [--configs top20 top40 top80 top160 topall] [--skip-preprocess]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("pipeline/imitation/case10/configs/data_volume_sweep")
SUMMARY_PATH = Path("data/output/experiment/case10_data_volume_sweep/summary.json")

DEFAULT_TAGS = ["top20", "top40", "top80", "top160", "topall", "top80_noswap"]
DEFAULT_OPPONENT = "baseline_v1"  # rulebase/case1
DEFAULT_EVAL_EPISODES = 20  # × P1/P2 swap → 40 matches per tag


def _run(cmd: list[str]) -> tuple[float, int, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return elapsed, proc.returncode, proc.stdout + proc.stderr


def _config_path(tag: str) -> Path:
    return CONFIG_DIR / f"il_case10_candidate_{tag}.yaml"


def _run_eval(tag: str, opponent: str, episodes: int) -> tuple[float, int, str]:
    """Run vs rulebase/case1 evaluation: `episodes` matches × P1/P2 swap."""
    return _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pipeline.imitation.case10.evaluation.eval_data_volume_sweep",
            "--tag",
            tag,
            "--opponent",
            opponent,
            "--episodes",
            str(episodes),
            "--swap-sides",
        ]
    )


def _run_one(
    tag: str,
    *,
    skip_preprocess: bool,
    skip_eval: bool,
    eval_opponent: str,
    eval_episodes: int,
) -> dict:
    cfg = _config_path(tag)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    record: dict = {"tag": tag, "config": str(cfg)}

    if not skip_preprocess:
        logger.info("[%s] preprocess start", tag)
        elapsed, rc, _ = _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "pipeline.imitation.case10.training.preprocess",
                "--config",
                str(cfg),
            ]
        )
        record["preprocess_seconds"] = elapsed
        record["preprocess_rc"] = rc
        if rc != 0:
            record["status"] = "preprocess_failed"
            return record

    logger.info("[%s] train start", tag)
    elapsed, rc, _ = _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pipeline.imitation.case10.training.train",
            "--config",
            str(cfg),
        ]
    )
    record["train_seconds"] = elapsed
    record["train_rc"] = rc
    if rc != 0:
        record["status"] = "train_failed"
        return record

    if skip_eval:
        record["status"] = "ok_no_eval"
        return record

    logger.info("[%s] eval vs %s (%d matches × swap) start", tag, eval_opponent, eval_episodes)
    elapsed, rc, _ = _run_eval(tag, eval_opponent, eval_episodes)
    record["eval_seconds"] = elapsed
    record["eval_rc"] = rc
    record["status"] = "ok" if rc == 0 else "eval_failed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-opponent", default=DEFAULT_OPPONENT)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for tag in args.configs:
        rec = _run_one(
            tag,
            skip_preprocess=args.skip_preprocess,
            skip_eval=args.skip_eval,
            eval_opponent=args.eval_opponent,
            eval_episodes=args.eval_episodes,
        )
        results.append(rec)
        SUMMARY_PATH.write_text(json.dumps(results, indent=2))
        logger.info("[%s] done: %s", tag, rec.get("status"))

    logger.info("summary written to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
