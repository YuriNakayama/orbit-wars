# Benchmarking & Tuning a JAX Port

The point of porting to JAX is **throughput on GPU**. A correct-but-slow port is
a failure. This file is the timing idiom and the tuning playbook — read it when
you reach the benchmark/tune step.

**Benchmark and tune on GPU only — never on the local CPU.** The dev laptop has
no CUDA GPU, and CPU wall-clock does not predict GPU behavior (the vmap-scaling
and memory characteristics that matter only appear on-device). Measurement and
tuning happen on a RunPod GPU pod via `dev/runpod`, using the
`bot/pipeline/_bench/<name>_gpu/run_bench.py` layout the onstart artifact
uploader already picks up (mirror `rollout_gpu` / `featurizer_gpu` /
`jax_env_gpu`). If no GPU pod is available right now, **do not run a CPU
benchmark as a stand-in and do not fabricate a number** — write the GPU bench
script, state that the figure is pending a RunPod run, and stop there.

Golden rule: **re-run the parity test after every tuning change.** Speed that
breaks correctness is not speed. A reordered reduction, an `f64→f32` switch, or
a `donate_argnums` mistake can silently shift outputs past the parity band.

## Table of contents

1. [Measuring honestly (on GPU)](#1-measuring-honestly-on-gpu)
2. [Apples-to-apples comparison](#2-apples-to-apples-comparison)
3. [The tuning playbook](#3-the-tuning-playbook)
4. [Where to put the benchmark](#4-where-to-put-the-benchmark)
5. [Reporting](#5-reporting)

---

## 1. Measuring honestly (on GPU)

Two mistakes produce fake numbers. Avoid both.

**Warm up — don't count compile time.** The first call to a jitted function
triggers XLA compilation (often seconds). Call once untimed to compile, then
time later calls.

**Block — don't time async dispatch.** JAX runs asynchronously: a jitted call
returns immediately and the compute happens later. If you stop the clock before
forcing completion you measure dispatch, not the kernel. Force completion with
`.block_until_ready()` (one output) or `jax.block_until_ready(pytree)` inside
the timed region. Use `time.perf_counter()` (mirroring `bench_jax_env.py`).

```python
import time, jax
runner = jax.jit(run)                      # run: state -> (final_state, outs)
final, _ = runner(state0); final.step.block_until_ready()   # warm-up, NOT timed
t0 = time.perf_counter()
for s in states:
    final, _ = runner(s); final.step.block_until_ready()    # block each iter
per_unit = (time.perf_counter() - t0) / len(states)
```

## 2. Apples-to-apples comparison

Run the Python/Torch original and the JAX version on the **identical workload**
(same horizon, episode/batch count, seed, reward/feature path) on the GPU pod.
Report each side's per-unit wall-clock and the ratio. The number that justifies
the port is **JAX-GPU per-episode at a realistic horizon (e.g. 500) vs the
Torch baseline**. Keep the Torch baseline cheap (e.g. `episodes=1`) if slow —
you need its per-unit cost, not a large sample.

### Batched vmap vs per-call dispatch — the result that decides everything

Measured on a real L4 (case2 intercept solver, this repo): the **batched** path —
one `vmap` over the whole (src×target) grid — beat the Python double loop by
**66× / 273× / 681×** at grids 8², 16², 24². The **per-call** path — the same
solver wired into the agent so each `plan_shot` makes its own jitted call — ran
the agent's `act()` at **0.18×, i.e. 5.6× *slower*** than pure Python, and was
*worse* on GPU than on CPU (0.18 vs 0.68). Same kernel, opposite verdict.

Why: a GPU call has high fixed dispatch + host↔device transfer latency. Amortized
over a 576-element vmap it's nothing; paid ~34 times per turn for individual
small calls it dominates, and GPU's per-call latency is *higher* than CPU's. So:

- **The speedup is in the batching, not the kernel.** Porting a function to JAX
  buys nothing — sometimes loses — unless the *caller* hands it the whole loop
  domain at once. Bench the batched grid AND the real per-call usage; report
  both. If your only number is the grid, you haven't shown the entry point got
  faster.
- A sequential, control-flow-driven consumer (a rule-based agent calling the
  solver per candidate, branching on each result) **cannot** be sped up by
  swapping in a JAX function call-for-call. It needs restructuring to
  "enumerate all candidates → one batched solve → select" — which may or may not
  be worth it. Say so honestly rather than reporting the grid number as if the
  agent got 600× faster.

## 3. The tuning playbook

Order: **profile → JIT & loops → memory & precision → parallelize → data
pipeline.** Biggest lever first; re-run parity after each change.

1. **Profile — measure, don't guess.** Find the bottleneck with
   `jax.profiler.trace` before changing anything. Beware async timing (block
   explicitly). Inspect the HLO (`jax.jit(f).lower(*args).compile().as_text()`)
   for stray `copy`/`transpose` or inter-device communication that snuck in.

2. **JIT optimization.** Wrap the *whole* hot loop in one `jit` to cut
   compile-boundary overhead. Fix shapes/config with `static_argnums` so varying
   input shape/dtype doesn't trigger recompiles (the per-seed recompile trap in
   `conversion-patterns.md` pattern 8 — it blew this repo's cache past 3 GB →
   SIGABRT). For large arrays overwritten each step, reuse buffers with
   `donate_argnums`.

3. **Loop structure.** Replace Python `for` with `lax.scan` (compile time drops
   by orders of magnitude — the body compiles once). Batch with `vmap`; nested
   `vmap` is fine. Consider `scan`'s `unroll` argument to trade compile time for
   per-step speed. **`vmap` batch size is the #1 GPU lever** — sweep it (e.g.
   {1, 16, 64}) and report the scaling curve, not a single point.

4. **Memory & precision.** Introduce mixed precision (`bfloat16`/`float16` for
   matmuls) while keeping accumulation in `float32`. For big models or long
   `scan`s, trade memory for recompute with `jax.checkpoint` (remat); use
   `remat_policy` to control which intermediates are saved. JAX defaults to f32 —
   only force f64 if parity genuinely requires it (f64 on GPU is far slower).

5. **Parallelize (multi-device).** When several devices are available, shard with
   `jax.sharding.NamedSharding` + `PartitionSpec`. Pin intermediate shardings
   with `with_sharding_constraint` and minimize collective communication
   (`psum`, etc.) frequency.

6. **Data pipeline.** Once compute is fast, data feeding becomes the bottleneck.
   Exploit `device_put`'s asynchrony to prefetch and cut host→device transfers.

Profile first, change one thing at a time, attribute each speedup, re-check
parity.

## 4. Where to put the benchmark

Real GPU benchmark → `bot/pipeline/_bench/<name>_gpu/run_bench.py`, mirroring the
existing `rollout_gpu` / `featurizer_gpu` / `jax_env_gpu` benches. These emit a
JSON result and run on RunPod (`dev/runpod`); the onstart artifact uploader picks
up anything under `_bench/<name>_gpu/` unchanged, so match that layout. A small
`typer` script like `bot/scripts/bench_jax_env.py` (with `--device gpu` and a
`--vmap N` sweep, warm-up + block built in) is fine for the bench body, but
**execute it on the GPU pod, not locally.**

## 5. Reporting

Give the user an actionable table — all rows measured on the GPU pod:

```
horizon=500, seed=0, device=RunPod RTX 3090
config                      per_ep     speedup vs torch baseline
torch (baseline)             4.50s      1.0x
jax   (gpu, vmap=1)          0.90s      5.0x
jax   (gpu, vmap=16)         0.25s     18.0x
jax   (gpu, vmap=64)         0.11s     41.0x
```

State the device, horizon/batch settings, the vmap-scaling curve, which tuning
changes you applied, and confirm parity still passed after tuning. If the GPU
pod isn't available, report "GPU bench pending RunPod" with the script ready —
never a CPU substitute or a guessed figure.
