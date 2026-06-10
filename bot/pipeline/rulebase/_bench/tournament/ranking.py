"""Minimal-cost strength RANKING of the JAX rule agents via adjacent verification.

The user established three facts that collapse a 21-pair round-robin into a
handful of comparisons:

  1. No rock-paper-scissors — strength is a strict TOTAL ORDER (transitive).
     So ranking is SORTING, and a verified order needs only n-1 adjacent
     comparisons, not C(n,2).
  2. Some agents are near-equal ("ほぼ強さの変わらない") — these must be MERGED
     into a tie bucket, not separated by burning more games.
  3. Case number ≈ strength — a STRONG PRIOR. The cheapest strategy is to VERIFY
     the prior's adjacent pairs; if every adjacent pair confirms it, transitivity
     gives the whole ranking for free.

Algorithm (insertion-sort against a prior):
  * Walk the prior order strong→weak. For each newly considered agent, compare it
    against the current tail of the confirmed sequence (its prior predecessor).
  * ``A_WINS`` (predecessor stronger) → append as a new lower rank.
  * ``TIE`` → merge into the predecessor's bucket (same rank).
  * ``B_WINS`` (INVERSION — the new agent beat its predecessor) → local repair:
    keep comparing it against earlier buckets (bubble up) until it loses or ties,
    then insert there. This is the only case that costs extra comparisons.

Each comparison is a fixed-N binary verdict from ``compare.compare_pair`` (no
SPRT early-stop). After EVERY comparison the running ranking is flushed to
``ranking.json`` + ``ranking.md`` and uploaded to S3 (crash-safe), mirroring
``run_tournament``'s observability.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from utils.gpu_bench import install_cuda_jax, reload_jax, run_dir

# Default prior, strongest → weakest (case number ≈ strength). jax_v9 (dormant,
# ≈v8) and jax_v7 (strict port incomplete — STAY/ACCUMULATE pending) are excluded.
DEFAULT_PRIOR: list[str] = ["jax_v8", "jax_v6", "jax_v4", "jax_v3", "jax_v2", "jax_v1"]
DEFAULT_HORIZON = 300  # ranking needs the win SIGN, not terminal-reward parity.
DEFAULT_SEEDS = 100  # per seat half → 200 games/comparison (fixed, no early-stop).

# Bound on inversion repair: a coherent prior should never bubble an agent past
# more than a few buckets. If it does, the prior is wrong in a non-adjacent way
# and the run aborts with a clear diagnostic rather than silently O(n^2)-ing.
MAX_REPAIR_HOPS = 5

# jax-binding symbols imported lazily post-reload (see run_tournament rationale).
compare_pair: Any = None
A_WINS: str = "a_wins"
B_WINS: str = "b_wins"
TIE: str = "tie"


def _bind_jax_symbols() -> None:
    """Import jax-binding symbols into module globals after cuda reload."""
    global compare_pair, A_WINS, B_WINS, TIE
    from pipeline.rulebase._bench.tournament.compare import (
        A_WINS as _A,
    )
    from pipeline.rulebase._bench.tournament.compare import (
        B_WINS as _B,
    )
    from pipeline.rulebase._bench.tournament.compare import (
        TIE as _T,
    )
    from pipeline.rulebase._bench.tournament.compare import (
        compare_pair as _cp,
    )

    compare_pair = _cp
    A_WINS, B_WINS, TIE = _A, _B, _T


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _s3_prefix() -> str | None:
    return os.environ.get("ORBIT_WARS_BEST_S3_PREFIX") or None


def _upload_s3(local: Path, prefix: str) -> None:
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(local), f"{prefix}/{local.name}"],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN s3 upload {local.name} failed: {exc}")


def _ranking_md(buckets: list[list[str]], comps: list[dict[str, Any]]) -> str:
    """Render buckets (rank groups, strongest first) + the comparison log."""
    lines = [
        f"# JAX rulebase strength ranking ({len(comps)} comparisons)",
        "",
        "Rank buckets — agents in the same bucket are statistically tied "
        "(Wilson 95% CI of win-rate straddles 0.5).",
        "",
        "| rank | agents |",
        "|---|---|",
    ]
    for i, bucket in enumerate(buckets, 1):
        lines.append(f"| {i} | {', '.join(bucket)} |")
    lines += [
        "",
        "## Comparison log",
        "",
        "| # | A | B | A_w | B_w | D | win% | 95% CI | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(comps, 1):
        ci = f"[{c['ci95_lo']:.3f}, {c['ci95_hi']:.3f}]"
        lines.append(
            f"| {i} | {c['a']} | {c['b']} | {c['a_wins']} | {c['b_wins']} | "
            f"{c['draws']} | {c['a_win_rate']:.3f} | {ci} | {c['decision']} |"
        )
    return "\n".join(lines) + "\n"


def _flush(out_dir: Path, payload: dict[str, Any], md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rj = out_dir / "ranking.json"
    rm = out_dir / "ranking.md"
    rj.write_text(json.dumps(payload, indent=2))
    rm.write_text(md)
    prefix = _s3_prefix()
    if prefix:
        _upload_s3(rj, prefix)
        _upload_s3(rm, prefix)


def _comp_record(res: Any) -> dict[str, Any]:
    """ComparisonResult → JSON-able dict for the log."""
    return {
        "a": res.a,
        "b": res.b,
        "a_wins": res.a_decisive_wins,
        "b_wins": res.b_decisive_wins,
        "draws": res.draws,
        "a_win_rate": res.a_win_rate,
        "ci95_lo": res.ci95_lo,
        "ci95_hi": res.ci95_hi,
        "decision": res.decision,
        "half_secs": list(res.half_secs),
    }


def rank_agents(
    prior: list[str],
    seeds: list[int],
    horizon: int,
    *,
    out_dir: Path,
    on_comparison: Any = None,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Insertion-sort ``prior`` (strong→weak) into verified rank buckets.

    ``on_comparison(result)`` is invoked after each comparison so the caller can
    flush incrementally. Returns (buckets, comparison_log).
    """
    if not prior:
        return [], []
    buckets: list[list[str]] = [[prior[0]]]
    comps: list[dict[str, Any]] = []

    def _run(a: str, b: str) -> Any:
        res = compare_pair(a, b, seeds, horizon)
        comps.append(_comp_record(res))
        if on_comparison is not None:
            on_comparison(buckets, comps)
        return res

    for agent in prior[1:]:
        # Start by comparing against the current weakest confirmed bucket.
        pos = len(buckets)  # candidate insertion index (below all buckets)
        hops = 0
        while pos > 0:
            rep = buckets[pos - 1][0]  # bucket representative (strongest-in-tie)
            res = _run(rep, agent)
            if res.decision == A_WINS:
                # rep stronger than agent → agent belongs strictly below this
                # bucket. Insert as its own new bucket at `pos`.
                buckets.insert(pos, [agent])
                break
            if res.decision == TIE:
                buckets[pos - 1].append(agent)  # near-equal → merge
                break
            # B_WINS: inversion — agent beat rep → bubble up one bucket.
            pos -= 1
            hops += 1
            if hops > MAX_REPAIR_HOPS:
                raise RuntimeError(
                    f"prior inversion exceeded {MAX_REPAIR_HOPS} repair hops at "
                    f"{agent}; prior order is wrong non-adjacently — re-examine."
                )
        else:
            # Bubbled past the top → agent is the new strongest.
            buckets.insert(0, [agent])

    return buckets, comps


def _load_config(path: str) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"ranking config at {path} must be a YAML mapping")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="YAML config path")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per seat half")
    ap.add_argument("--horizon", type=int, default=None, help="turns/game")
    ap.add_argument("--out", type=str, default="", help="run_dir for artifacts")
    ap.add_argument("--prior", type=str, default="", help="comma list strong→weak")
    ap.add_argument("--cuda-jax", action="store_true", help="install jax[cuda12]")
    args = ap.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    seeds_n = (
        args.seeds if args.seeds is not None else int(cfg.get("seeds", DEFAULT_SEEDS))
    )
    horizon = (
        args.horizon
        if args.horizon is not None
        else int(cfg.get("horizon", DEFAULT_HORIZON))
    )
    cuda_jax = args.cuda_jax or bool(cfg.get("cuda_jax", False))
    prior_spec = args.prior or cfg.get("prior", "")

    if cuda_jax:
        install_cuda_jax()
    reload_jax(extra_prefixes=("pipeline.rulebase",))
    _bind_jax_symbols()

    if isinstance(prior_spec, list):
        prior = [str(a).strip() for a in prior_spec if str(a).strip()]
    elif prior_spec:
        prior = [s.strip() for s in str(prior_spec).split(",") if s.strip()]
    else:
        prior = list(DEFAULT_PRIOR)

    out_dir = Path(args.out) if args.out else run_dir()
    seeds = list(range(seeds_n))
    started = time.time()

    _log(f"ranking: prior={prior}, {seeds_n} seeds/half, horizon={horizon}")

    payload: dict[str, Any] = {
        "status": "running",
        "prior": prior,
        "seeds_per_half": seeds_n,
        "horizon": horizon,
        "buckets": [],
        "comparisons": [],
    }
    _flush(out_dir, payload, _ranking_md([], []))

    def _on_comparison(buckets: list[list[str]], comps: list[dict[str, Any]]) -> None:
        payload["buckets"] = buckets
        payload["comparisons"] = comps
        payload["elapsed_secs"] = round(time.time() - started, 1)
        _flush(out_dir, payload, _ranking_md(buckets, comps))
        last = comps[-1]
        _log(
            f"  cmp {len(comps)}: {last['a']} vs {last['b']} → {last['decision']} "
            f"({last['a_wins']}-{last['b_wins']}, {last['half_secs'][0]}+"
            f"{last['half_secs'][1]}s)"
        )

    try:
        buckets, comps = rank_agents(
            prior, seeds, horizon, out_dir=out_dir, on_comparison=_on_comparison
        )
        payload["status"] = "completed"
        payload["buckets"] = buckets
        payload["comparisons"] = comps
        payload["elapsed_secs"] = round(time.time() - started, 1)
        _flush(out_dir, payload, _ranking_md(buckets, comps))
        _log(
            f"ranking complete in {payload['elapsed_secs']}s "
            f"({len(comps)} comparisons)"
        )
        _log("\n" + _ranking_md(buckets, comps))
        return 0
    except Exception:  # noqa: BLE001 — crash-safety: persist failure + traceback.
        tb = traceback.format_exc()
        payload["status"] = "failed"
        payload["error"] = tb
        payload["elapsed_secs"] = round(time.time() - started, 1)
        _flush(
            out_dir,
            payload,
            _ranking_md(payload["buckets"], payload["comparisons"]),
        )
        _log(f"RANKING FAILED:\n{tb}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
