"""GPU verification bench for case8 compute_actions (XLA compile + runtime).

Answers two questions on real GPU hardware:
  1. compile time  — does compute_actions XLA-compile in minutes (not the
     documented ~2h CPU hang)?
  2. runtime       — does GPU vmap parallelism reduce the CPU ~24s/call to a
     practical figure (single call + batched vmap over N states)?

Run on a GPU pod:  python -m pipeline.rulebase.case8.baseline_jax._gpu_bench
Prints a JSON line so the host can scrape it via base64/stdout.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from .agent_jax import _modes_from_features, compute_actions
from .world_features import build_world_features_from_state


def _state_at(seed: int, advance: int):
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    st = reset(seed=seed, num_agents=2)
    for _ in range(advance):
        st, _, t = step(st, noop)
        if bool(t):
            break
    return st


def _fm(seed: int, advance: int):
    st = _state_at(seed, advance)
    feat = build_world_features_from_state(st, seat=1)
    modes = _modes_from_features(feat)
    return feat, modes


def main() -> None:
    out: dict[str, object] = {"platform": str(jax.devices()[0].platform),
                              "device": str(jax.devices()[0])}

    feat0, modes0 = _fm(0, 12)

    # (1) compile time: jit + block on first result
    fn = jax.jit(compute_actions)
    t0 = time.perf_counter()
    r0 = fn(feat0, modes0)
    r0.block_until_ready()
    out["compile_s"] = round(time.perf_counter() - t0, 2)

    # (2a) warm single-call runtime (compiled trace reused, new inputs)
    feat1, modes1 = _fm(3, 30)
    # warm with same shapes
    fn(feat1, modes1).block_until_ready()
    n = 10
    t0 = time.perf_counter()
    for s in range(n):
        f, m = _fm(s, (s % 6) * 4)
        fn(f, m).block_until_ready()
    out["warm_single_ms"] = round((time.perf_counter() - t0) / n * 1000, 1)

    # (3) batched vmap over B states — the GPU-parallelism question
    for B in (8, 64):
        feats = [_fm(s, (s % 6) * 4) for s in range(B)]
        stacked_f = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[fm[0] for fm in feats]
        )
        stacked_m = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[fm[1] for fm in feats]
        )
        vfn = jax.jit(jax.vmap(compute_actions))
        tc = time.perf_counter()
        vfn(stacked_f, stacked_m).block_until_ready()
        out[f"vmap{B}_compile_s"] = round(time.perf_counter() - tc, 2)
        tr = time.perf_counter()
        for _ in range(5):
            vfn(stacked_f, stacked_m).block_until_ready()
        per_batch = (time.perf_counter() - tr) / 5
        out[f"vmap{B}_batch_ms"] = round(per_batch * 1000, 1)
        out[f"vmap{B}_per_state_ms"] = round(per_batch / B * 1000, 2)

    print("BENCH_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
