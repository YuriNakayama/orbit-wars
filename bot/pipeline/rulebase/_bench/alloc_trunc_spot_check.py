"""Spot identity check: allocator truncation K=256/64 vs full scan (CPU).

Quick pre-GPU gate for the re-applied MAX_ALLOC_CANDIDATES truncation. The full
regression lives in tests/e2e/.../test_strict_jax_identity.py; this is the fast
inline version (12 comparisons) run before committing / launching the GPU smoke.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, NUM_AGENTS_MAX
from orbit_wars_jax.step import step as env_step

import pipeline.rulebase.case1.baseline_jax.strict.allocator_jax as al
from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import (
    _modes_from_features,
    compute_actions,
)
from pipeline.rulebase.case1.baseline_jax.strict.world_features import (
    build_world_features_from_state,
)


def _actions_k(feats, modes, k: int) -> np.ndarray:
    orig = al.MAX_ALLOC_CANDIDATES
    al.MAX_ALLOC_CANDIDATES = k
    try:
        out = compute_actions(feats, modes)
        jax.block_until_ready(out)
        return np.asarray(out)
    finally:
        al.MAX_ALLOC_CANDIDATES = orig


def main() -> None:
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    mism = 0
    checked = 0
    for s in range(6):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 3) * 12):
            state, _, term = env_step(state, noop)
            if bool(term):
                break
        for seat in (0, 1):
            feats = build_world_features_from_state(state, seat)
            modes = _modes_from_features(feats)
            ref = _actions_k(feats, modes, 4608)
            for k in (256, 64):
                got = _actions_k(feats, modes, k)
                ok = (
                    np.array_equal(got[:, 0], ref[:, 0])
                    and np.array_equal(got[:, 2], ref[:, 2])
                    and np.allclose(got[:, 1], ref[:, 1], atol=1e-3)
                )
                if not ok:
                    mism += 1
                    print(f"MISMATCH K={k} seed={s} seat={seat}")
                checked += 1
    verdict = "PASS" if mism == 0 else "FAIL"
    print(f"spot identity: checked={checked} mism={mism} {verdict}")


if __name__ == "__main__":
    main()
