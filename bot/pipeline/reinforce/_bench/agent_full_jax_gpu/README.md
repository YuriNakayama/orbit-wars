# agent_full_jax_gpu — GPU vmap throughput bench

Measures the vmapped JAX-vs-JAX self-play throughput (env-steps/sec) of the
faithful rulebase/case1 JAX agent (`core_jax.agent_full_jax`) across batch sizes,
versus a single Python `baseline_v1` game. This is the headline number that
justifies the rulebase→JAX port (JAX wins throughput via GPU batch parallelism).

## Local CPU smoke (fast, no GPU)

```bash
cd bot
ORBIT_WARS_RUN_DIR=/tmp/b uv run python -m \
  pipeline.reinforce._bench.agent_full_jax_gpu.run_bench
cat /tmp/b/agent_full_jax_bench.json
```

CPU reference (measured 2026-06-03, M-series, B∈{1,8}):
- `python_v1_single`: ~44 env-steps/s
- `jax_vmap` B=1: ~5/s, B=8: ~6/s  (CPU: JAX loses on latency, expected —
  GPU batch parallelism is the whole point; B sweep flat on CPU = core-bound.)

## GPU run (RunPod) — the headline figure

```bash
git push origin <branch>
dev/runpod train "$(git rev-parse HEAD)" --case bench_agent_full_jax_gpu --watch
dev/runpod pull <run_id>        # fetches agent_full_jax_bench.json
```

### Known blocker (2026-06-03): network-volume datacenter constraint

`dev/runpod train` always attaches the shared `orbit_wars` network volume
(SECURE cloud, fixed DC). When that DC has no matching GPU offer (Low-stock
3090/A6000/4090, or H100 above `--max-dph`), launch fails with
`No offers matched` even though `dev/runpod stock` shows capacity elsewhere.

This bench writes only a small JSON (no volume needed). To run it when the
volume DC is GPU-starved, use one of:
- raise `--max-dph` and target a High-stock GPU **in the volume's DC**, or
- `dev/runpod dev <sha> --case bench_agent_full_jax_gpu` (interactive, volume
  optional) then `dev/runpod exec <run_id> -- python -m
  pipeline.reinforce._bench.agent_full_jax_gpu.run_bench`, then
  `dev/runpod destroy <run_id>` (interactive pods bill until destroyed).

The bench code itself is verified (RunPod preflight: import + CPU smoke OK).
Only execution is gated on RunPod capacity/volume-region.
