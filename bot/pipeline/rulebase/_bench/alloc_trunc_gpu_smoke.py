"""GPU smoke: per-turn wall-clock vs MAX_ALLOC_CANDIDATES (K) and batch size.

Validates the Phase-1 hypothesis: the GPU per-turn cost (~18s at any batch) is
the SEQUENTIAL kernel-launch chain of the allocator's N=4608-step lax.scan, so
truncating to K=256/64 (identity-preserving — the tail is -inf no-ops) should
cut per-turn roughly in proportion to the removed sequential steps.

Run ON the GPU pod (interactive):
    python -m pipeline.rulebase._bench.alloc_trunc_gpu_smoke

Prints one line per (K, batch) config with median per-turn seconds, flushed
incrementally so a `tail -f` shows live progress. K is a module constant read at
trace time, so each config re-jits the turn step (fresh compile per config).
"""

from __future__ import annotations

import time

from utils.gpu_bench import install_cuda_jax, reload_jax


def main() -> None:
    # cuda12 wheel BEFORE any jax import (abseil SetTimeZone crash otherwise).
    install_cuda_jax()
    reload_jax(extra_prefixes=("pipeline.rulebase",))

    import jax
    import jax.numpy as jnp
    from orbit_wars_jax.reset import reset
    from orbit_wars_jax.step import NUM_AGENTS_MAX

    import pipeline.rulebase.case1.baseline_jax.strict.allocator_jax as al
    from pipeline.rulebase._bench.tournament.agents_jax import action_fn
    from pipeline.rulebase._bench.tournament.selfplay_host import _make_turn_step

    print(f"devices: {jax.devices()}", flush=True)

    # (K, batch, timed_turns). Full-scan baseline kept short (known ~18s/turn);
    # truncated configs get more turns for a stable median.
    configs = [
        (4608, 8, 2),
        (256, 8, 4),
        (64, 8, 4),
        (256, 64, 4),
        (64, 64, 4),
    ]

    results: list[tuple[int, int, float]] = []
    for k, batch, turns in configs:
        al.MAX_ALLOC_CANDIDATES = k
        seeds = list(range(batch))
        states = [reset(seed=s, num_agents=2) for s in seeds]
        batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
        done = jnp.zeros((batch,), dtype=jnp.bool_)
        final_rewards = jnp.zeros((batch, NUM_AGENTS_MAX), dtype=jnp.float32)
        tcount = jnp.zeros((batch,), dtype=jnp.int32)

        # Fresh jit per config so the current K is baked into the trace.
        turn_step = _make_turn_step(action_fn("jax_v1", 0), action_fn("jax_v1", 1))

        t0 = time.perf_counter()
        batched, done, final_rewards, tcount = turn_step(
            batched, done, final_rewards, tcount
        )
        done.block_until_ready()
        compile_s = time.perf_counter() - t0
        print(f"K={k:5d} batch={batch:3d} compile+1st_turn={compile_s:7.1f}s", flush=True)

        per_turn: list[float] = []
        for _ in range(turns):
            t0 = time.perf_counter()
            batched, done, final_rewards, tcount = turn_step(
                batched, done, final_rewards, tcount
            )
            done.block_until_ready()
            per_turn.append(time.perf_counter() - t0)
        med = sorted(per_turn)[len(per_turn) // 2]
        results.append((k, batch, med))
        print(
            f"K={k:5d} batch={batch:3d} per_turn={med:7.2f}s "
            f"(all: {[round(t, 2) for t in per_turn]})",
            flush=True,
        )

    print("\n=== SUMMARY ===", flush=True)
    base = next((m for kk, bb, m in results if kk == 4608), None)
    for k, batch, med in results:
        ratio = f"{base / med:5.1f}x" if base and k != 4608 else "  1.0x"
        print(f"K={k:5d} batch={batch:3d} per_turn={med:7.2f}s speedup={ratio}", flush=True)


if __name__ == "__main__":
    main()
