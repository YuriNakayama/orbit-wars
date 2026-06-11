"""Probe: measure live-source counts + verify A/B grid-speedup identity & speed.

Run under bot/ (worktree on PYTHONPATH). CPU is fine for identity (small); speed
delta is compile+run wall-clock on whatever device is present.

Plan A (mask): keep the 48x48 vmap, but force dead source ROWS to -inf/invalid
              BEFORE flatten — tests whether masking alone changes the output
              (it must not) and whether XLA elides anything (it won't structurally).
Plan B (gather): vmap per_src only over the first K live source slots, scatter the
              rows back to their true slot s in a -inf-initialized (48,48) grid.
              Tests byte-identity vs the full grid when K >= live count.

Outputs: live-source histogram (drives MAX_LIVE_SRC), and identity verdicts.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import EnvState
from orbit_wars_jax.step import step as env_step

from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import (
    _modes_from_features,
)
from pipeline.rulebase.case1.baseline_jax.strict.missions_capture_jax import (
    _capture_cell,
    build_capture_grid,
)
from pipeline.rulebase.case1.baseline_jax.strict.world_features import (
    build_world_features_from_state,
)

MAX_PLANETS = 48


def _advance(state: EnvState, turns: int) -> EnvState:
    """Step forward with BOTH seats playing the real rule agent (captures planets,
    so ownership grows — needed to find the true peak live-source count)."""
    from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, NUM_AGENTS_MAX

    from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import compute_actions

    def act(st: EnvState, seat: int) -> jax.Array:
        f = build_world_features_from_state(st, seat)
        return compute_actions(f, _modes_from_features(f))

    peak = 0
    for _ in range(turns):
        a0 = act(state, 0)
        a1 = act(state, 1)
        env_actions = (
            jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, jnp.float32)
            .at[0]
            .set(a0)
            .at[1]
            .set(a1)
        )
        state, _r, term = env_step(state, env_actions)
        f0 = build_world_features_from_state(state, 0)
        f1 = build_world_features_from_state(state, 1)
        peak = max(peak, _live_source_count(f0), _live_source_count(f1))
        if bool(term):
            break
    _advance.peak = peak  # type: ignore[attr-defined]
    return state


def _live_source_count(features) -> int:
    mine = features.planet_valid & (features.owner == features.player)
    return int(jnp.sum(mine & (features.available > 0)))


def _grid_eq(a, b) -> bool:
    """Byte-identity of two CaptureGrid NamedTuples (valid + score + key fields)."""
    for fa, fb in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        if fa.shape != fb.shape:
            return False
        # NaN-safe exact compare (score uses -inf/0; ints exact).
        if not bool(jnp.all((fa == fb) | (jnp.isnan(fa) & jnp.isnan(fb)))):
            return False
    return True


def _build_b_grid(features, modes, k: int):
    """Plan B: vmap per_src over first K live source slots, scatter to true slots."""
    idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    mine = features.planet_valid & (features.owner == features.player)
    live = mine & (features.available > 0)
    # First K live source slot indices (padded with 0 where fewer than K).
    order = jnp.argsort(~live)  # live (False=~live small) first, stable by index
    live_slots = order[:k]

    def per_src(s):
        def per_tgt(t):
            return _capture_cell(features, modes, *_timelines(features), s, t)

        return jax.vmap(per_tgt)(idx)

    rows = jax.vmap(per_src)(live_slots)  # (K, 48) CaptureGrid

    # Reference full grid to get the -inf/invalid template, then scatter live rows.
    full = build_capture_grid(features, modes)
    # Overwrite only the live rows at their true slot; dead rows stay as full's
    # (which are already invalid/-inf for dead sources — identical content).
    return jax.tree_util.tree_map(
        lambda f_full, f_rows: f_full.at[live_slots].set(f_rows), full, rows
    )


def _timelines(features):
    from pipeline.rulebase.case1.baseline_jax.strict.missions_capture_jax import (
        _base_timelines,
    )

    return _base_timelines(features)


def main() -> None:
    # Spread of states: fresh + advanced to mid/late game for varied ownership.
    states = []
    peaks = []
    for s in range(4):
        st = _advance(reset(seed=s, num_agents=2), 120)
        states.append(st)
        peaks.append(getattr(_advance, "peak", 0))
    print(f"PEAK live-source per game (rule-agent self-play, 120t): {peaks}")
    print(f"  GLOBAL PEAK = {max(peaks)}  -> MAX_LIVE_SRC must be >= this + margin")

    feats = [build_world_features_from_state(st, 0) for st in states]
    counts = [_live_source_count(f) for f in feats]
    print(f"final-state live-source counts: {counts}")

    f0 = feats[0]
    m0 = _modes_from_features(f0)

    print("\n=== speed: full grid build (compile + run) ===")
    t0 = time.perf_counter()
    full = build_capture_grid(f0, m0)
    jax.block_until_ready(full)
    print(f"  full: {time.perf_counter() - t0:.2f}s")

    print("\n=== identity: plan B (gather K live, scatter) vs full ===")
    k_safe = max(counts) + 2
    for f in feats[:4]:
        m = _modes_from_features(f)
        full_g = build_capture_grid(f, m)
        b_g = _build_b_grid(f, m, k_safe)
        ok = _grid_eq(full_g, b_g)
        print(
            f"  state live={_live_source_count(f)} K={k_safe}:"
            f" identity={'OK' if ok else 'MISMATCH'}"
        )


if __name__ == "__main__":
    main()
