"""Round-robin tournament between the in-JAX rule agents → win-rate leaderboard.

Plays every unordered pair of agents over N seeds in BOTH seat assignments
(removing the env's seat-0/seat-1 start asymmetry), aggregates per-agent
win/loss/draw, and ranks by win-rate with a Wilson 95 % CI.

Observability (RunPod requirements):
  * incremental: after EACH pair finishes, the running standings + the pairwise
    matrix are written to ``tournament.json`` (run_dir) and the human-readable
    ``leaderboard.md`` — visible locally via ``dev/runpod tail`` / S3 mid-run.
  * crash-safe: every incremental write is also uploaded to S3 immediately
    (``ORBIT_WARS_BEST_S3_PREFIX``), so a mid-run failure keeps all completed
    pairs + a ``status: failed`` record with the traceback.
  * phase timing: per-pair compile + play wall-clock recorded in the matrix; the
    onstart nvidia-smi / system_monitor samplers cover GPU/CPU/mem.

Usage (on pod or locally):
    python -m pipeline.rulebase._bench.tournament.run_tournament \
        --seeds 300 --out data/output/.../runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from evaluate.vs_baseline import wilson_ci
from pipeline.rulebase._bench.tournament.agents_jax import action_fn, list_agents
from pipeline.rulebase._bench.tournament.selfplay_host import (
    OUTCOME_DRAW,
    OUTCOME_SEAT0_WIN,
    OUTCOME_SEAT1_WIN,
    run_host_batch,
)
from utils.gpu_bench import install_cuda_jax, reload_jax, run_dir

HORIZON = 500  # must be 500 — games run ~497 turns; <500 drops terminal reward.


def _log(msg: str) -> None:
    """Timestamped line to stdout (captured to train.log, tailed live)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _s3_prefix() -> str | None:
    return os.environ.get("ORBIT_WARS_BEST_S3_PREFIX") or None


def _upload_s3(local: Path, prefix: str) -> None:
    """Best-effort immediate S3 upload (crash-safety); never raises."""
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(local), f"{prefix}/{local.name}"],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN s3 upload {local.name} failed: {exc}")


def _flush(out_dir: Path, payload: dict[str, Any], leaderboard_md: str) -> None:
    """Write tournament.json + leaderboard.md to run_dir and (if set) S3."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tj = out_dir / "tournament.json"
    lb = out_dir / "leaderboard.md"
    tj.write_text(json.dumps(payload, indent=2))
    lb.write_text(leaderboard_md)
    prefix = _s3_prefix()
    if prefix:
        _upload_s3(tj, prefix)
        _upload_s3(lb, prefix)


def _standings(records: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Per-agent win/loss/draw → win-rate + Wilson CI, sorted by win-rate desc."""
    rows: list[dict[str, Any]] = []
    for name, c in records.items():
        games = c["wins"] + c["losses"] + c["draws"]
        win_rate = (c["wins"] / games) if games else 0.0
        lo, hi = wilson_ci(c["wins"], games)
        rows.append(
            {
                "agent": name,
                "games": games,
                "wins": c["wins"],
                "losses": c["losses"],
                "draws": c["draws"],
                "win_rate": win_rate,
                "win_rate_ci95_lo": lo,
                "win_rate_ci95_hi": hi,
            }
        )
    return sorted(rows, key=lambda r: -float(r["win_rate"]))


def _leaderboard_md(rows: list[dict[str, Any]], done: int, total: int) -> str:
    lines = [
        f"# JAX rulebase tournament leaderboard ({done}/{total} pairs done)",
        "",
        "| rank | agent | games | win% | 95% CI | W | L | D |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        ci = f"[{r['win_rate_ci95_lo']:.3f}, {r['win_rate_ci95_hi']:.3f}]"
        lines.append(
            f"| {i} | {r['agent']} | {r['games']} | {r['win_rate']:.3f} | {ci} "
            f"| {r['wins']} | {r['losses']} | {r['draws']} |"
        )
    return "\n".join(lines) + "\n"


def _play_pair(
    a: str, b: str, seeds: list[int], horizon: int
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Play a-vs-b over `seeds` in BOTH seat assignments.

    Returns (a_record, b_record, timing) where each record is wins/losses/draws.
    Seat-swapped halves cancel the env's seat-start asymmetry.
    """
    a_rec = {"wins": 0, "losses": 0, "draws": 0}
    b_rec = {"wins": 0, "losses": 0, "draws": 0}
    timing: dict[str, float] = {}

    def _tally(outcome: jax.Array, seat0: str, seat1: str) -> None:
        s0 = int(jnp.sum(outcome == OUTCOME_SEAT0_WIN))
        s1 = int(jnp.sum(outcome == OUTCOME_SEAT1_WIN))
        d = int(jnp.sum(outcome == OUTCOME_DRAW))
        rec0 = a_rec if seat0 == a else b_rec
        rec1 = a_rec if seat1 == a else b_rec
        rec0["wins"] += s0
        rec0["losses"] += s1
        rec1["wins"] += s1
        rec1["losses"] += s0
        rec0["draws"] += d
        rec1["draws"] += d

    # Half 1: a=seat0, b=seat1.   Half 2: b=seat0, a=seat1 (seat-swap).
    for half, (s0_name, s1_name) in enumerate(((a, b), (b, a)), 1):
        t0 = time.perf_counter()
        res = run_host_batch(
            seeds,
            action_fn(s0_name, 0),
            action_fn(s1_name, 1),
            horizon=horizon,
        )
        outcome = res["outcome"]
        outcome.block_until_ready()
        timing[f"half{half}_secs"] = round(time.perf_counter() - t0, 2)
        _tally(outcome, s0_name, s1_name)

    return a_rec, b_rec, timing


def _load_config(path: str) -> dict[str, Any]:
    """Load a tournament YAML config (seeds/horizon/agents/cuda_jax)."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"tournament config at {path} must be a YAML mapping")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    # The RunPod onstart only allows ``--config <path>`` (config_arg validator),
    # so the pod path passes a YAML; CLI flags override it for local use.
    ap.add_argument("--config", type=str, default="", help="YAML config path")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per seat half")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="run_dir for artifacts (default: $ORBIT_WARS_RUN_DIR via run_dir())",
    )
    ap.add_argument(
        "--horizon", type=int, default=None, help="turns/game (must be 500)"
    )
    ap.add_argument(
        "--agents", type=str, default="", help="comma list (default: all 7)"
    )
    ap.add_argument(
        "--cuda-jax",
        action="store_true",
        help="install jax[cuda12] on the pod before running (GPU)",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    seeds_n = args.seeds if args.seeds is not None else int(cfg.get("seeds", 300))
    horizon = (
        args.horizon if args.horizon is not None else int(cfg.get("horizon", HORIZON))
    )
    cuda_jax = args.cuda_jax or bool(cfg.get("cuda_jax", False))
    agents_spec = args.agents or cfg.get("agents", "")

    if cuda_jax:
        install_cuda_jax()
        reload_jax()

    if isinstance(agents_spec, list):
        agents = [str(a).strip() for a in agents_spec if str(a).strip()]
    elif agents_spec:
        agents = [s.strip() for s in str(agents_spec).split(",") if s.strip()]
    else:
        agents = list_agents()
    out_dir = Path(args.out) if args.out else run_dir()
    seeds = list(range(seeds_n))
    pairs = list(combinations(agents, 2))
    records: dict[str, dict[str, int]] = {
        a: {"wins": 0, "losses": 0, "draws": 0} for a in agents
    }
    matrix: list[dict[str, Any]] = []
    started = time.time()

    _log(f"tournament: {len(agents)} agents, {len(pairs)} pairs, {seeds_n} seeds/half")
    _log(f"agents: {agents}")

    payload: dict[str, Any] = {
        "status": "running",
        "agents": agents,
        "seeds_per_half": seeds_n,
        "horizon": horizon,
        "total_pairs": len(pairs),
        "pairs_done": 0,
        "matrix": matrix,
        "standings": [],
    }
    _flush(out_dir, payload, _leaderboard_md([], 0, len(pairs)))

    try:
        for idx, (a, b) in enumerate(pairs, 1):
            _log(f"pair {idx}/{len(pairs)}: {a} vs {b} ...")
            t0 = time.perf_counter()
            a_rec, b_rec, timing = _play_pair(a, b, seeds, horizon)
            pair_secs = round(time.perf_counter() - t0, 2)
            for k in ("wins", "losses", "draws"):
                records[a][k] += a_rec[k]
                records[b][k] += b_rec[k]
            matrix.append(
                {
                    "a": a,
                    "b": b,
                    "a_wins": a_rec["wins"],
                    "b_wins": b_rec["wins"],
                    "draws": a_rec["draws"],
                    "pair_secs": pair_secs,
                    **timing,
                }
            )
            payload["pairs_done"] = idx
            payload["standings"] = _standings(records)
            payload["elapsed_secs"] = round(time.time() - started, 1)
            _flush(
                out_dir, payload, _leaderboard_md(payload["standings"], idx, len(pairs))
            )
            _log(
                f"  done in {pair_secs}s — {a} {a_rec['wins']}W / {b} {b_rec['wins']}W"
                f" / {a_rec['draws']}D"
            )

        payload["status"] = "completed"
        payload["elapsed_secs"] = round(time.time() - started, 1)
        _flush(
            out_dir,
            payload,
            _leaderboard_md(payload["standings"], len(pairs), len(pairs)),
        )
        _log(f"tournament complete in {payload['elapsed_secs']}s")
        _log("\n" + _leaderboard_md(payload["standings"], len(pairs), len(pairs)))
        return 0
    except Exception:  # noqa: BLE001 — crash-safety: persist failure + traceback.
        tb = traceback.format_exc()
        payload["status"] = "failed"
        payload["error"] = tb
        payload["elapsed_secs"] = round(time.time() - started, 1)
        _flush(
            out_dir,
            payload,
            _leaderboard_md(
                payload.get("standings", []), payload["pairs_done"], len(pairs)
            ),
        )
        _log(f"TOURNAMENT FAILED after {payload['pairs_done']} pairs:\n{tb}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
