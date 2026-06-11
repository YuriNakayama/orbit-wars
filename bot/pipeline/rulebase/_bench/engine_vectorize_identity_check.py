"""Identity gate: case8 aim vectorizations (first-hit + intercept-search).

The two aim scans were vectorized (scan -> vmap+argmin/argmax) to collapse the
219x110 nested sequential chain. This script re-implements the ORIGINAL scan
versions verbatim, monkeypatches them into aim_jax, and byte-compares
compute_actions old-vs-new across real game states (src id + ship count exact,
angle atol 1e-3).

Run:  uv run python pipeline/rulebase/_bench/engine_vectorize_identity_check.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from orbit_wars_jax.reset import reset  # noqa: E402
from orbit_wars_jax.step import (  # noqa: E402
    MAX_LAUNCHES_PER_AGENT,
    NUM_AGENTS_MAX,
)
from orbit_wars_jax.step import step as env_step  # noqa: E402

import pipeline.rulebase.case8.baseline_jax.aim_jax as aim  # noqa: E402
from pipeline.rulebase.case8.baseline_jax.agent_jax import (  # noqa: E402
    _modes_from_features,
    compute_actions,
)
from pipeline.rulebase.case8.baseline_jax.world_features import (  # noqa: E402
    build_world_features_from_state,
)

_BIG = jnp.float32(1e18)


def _old_first_engine_hit_turn_jax(
    sx, sy, sr, angle, ships, init_x, init_y, init_r, tcx, tcy, tr,
    ang_vel, path, path_index, path_len, turn_lo, turn_hi,
):  # ORIGINAL scan version (pre-vectorization), verbatim.
    speed = aim.fleet_speed_jax(jnp.maximum(jnp.int32(1), ships))
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    offset = sr + aim._LAUNCH_OFFSET
    start_x = sx + cos_a * offset
    start_y = sy + sin_a * offset
    start = jnp.maximum(jnp.int32(1), turn_lo)

    def fleet_at(t):
        tf = t.astype(jnp.float32)
        return start_x + cos_a * speed * tf, start_y + sin_a * speed * tf

    def body(carry, t):
        found, hit_turn = carry
        active = (t >= start) & (t <= turn_hi)
        fx, fy = fleet_at(t)
        fpx, fpy = fleet_at(t - 1)
        px_now, py_now, ok_now = aim._predict_target_position_jax(
            t, tcx, tcy, init_x, init_y, init_r, ang_vel, path, path_index, path_len
        )
        px_prev, py_prev, ok_prev = aim._predict_target_position_jax(
            t - 1, tcx, tcy, init_x, init_y, init_r, ang_vel, path, path_index, path_len
        )
        hit = aim._swept_pair_hit_jax(
            fpx, fpy, fx, fy, px_prev, py_prev, px_now, py_now, tr
        )
        is_hit = active & ok_now & ok_prev & hit
        take = is_hit & (~found)
        return (found | take, jnp.where(take, t, hit_turn)), None

    (found, hit_turn), _ = jax.lax.scan(
        body, (jnp.bool_(False), jnp.int32(0)), aim._TURN_GRID
    )
    return hit_turn, found


def _old_search_safe_intercept_jax(
    sx, sy, sr, ships, init_x, init_y, init_r, tcx, tcy, tr,
    ang_vel, path, path_index, path_len, max_turns,
):  # ORIGINAL scan version (pre-vectorization), verbatim.
    def body(carry, cand):
        best_h, best_d, best_angle, best_turn, best_ix, best_iy = carry
        in_range = cand <= max_turns.astype(jnp.float32)
        lo_x, lo_y, lo_ok = aim._predict_target_position_jax(
            jnp.floor(cand), tcx, tcy, init_x, init_y, init_r,
            ang_vel, path, path_index, path_len,
        )
        frac = cand - jnp.floor(cand)
        hi_x, hi_y, hi_ok = aim._predict_target_position_jax(
            jnp.floor(cand) + 1.0, tcx, tcy, init_x, init_y, init_r,
            ang_vel, path, path_index, path_len,
        )
        use_interp = (frac > 1e-9) & hi_ok
        lead_x = jnp.where(use_interp, lo_x + (hi_x - lo_x) * frac, lo_x)
        lead_y = jnp.where(use_interp, lo_y + (hi_y - lo_y) * frac, lo_y)
        angle, hit_turn, hit_ok = aim._hit_turn_for_target_position_jax(
            sx, sy, sr, lead_x, lead_y, ships, init_x, init_y, init_r,
            tcx, tcy, tr, ang_vel, path, path_index, path_len, max_turns,
        )
        ax, ay, actual_ok = aim._predict_target_position_jax(
            hit_turn.astype(jnp.float32), tcx, tcy, init_x, init_y, init_r,
            ang_vel, path, path_index, path_len,
        )
        valid = in_range & lo_ok & hit_ok & actual_ok
        score_h = jnp.where(valid, hit_turn, jnp.int32(2**30))
        score_d = jnp.where(valid, jnp.abs(hit_turn.astype(jnp.float32) - cand), _BIG)
        better = (score_h < best_h) | ((score_h == best_h) & (score_d < best_d))
        take = valid & better
        return (
            jnp.where(take, score_h, best_h),
            jnp.where(take, score_d, best_d),
            jnp.where(take, angle, best_angle),
            jnp.where(take, hit_turn, best_turn),
            jnp.where(take, ax, best_ix),
            jnp.where(take, ay, best_iy),
        ), None

    init = (
        jnp.int32(2**30), jnp.float32(_BIG), jnp.float32(0.0),
        jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0),
    )
    (best_h, _bd, best_angle, best_turn, best_ix, best_iy), _ = jax.lax.scan(
        body, init, aim._CANDIDATE_TURNS
    )
    valid = (best_h < 2**30) & (max_turns > 0)
    return best_angle, best_turn, best_ix, best_iy, valid


def main() -> None:
    new_fns = (aim._first_engine_hit_turn_jax, aim._search_safe_intercept_jax)
    old_fns = (_old_first_engine_hit_turn_jax, _old_search_safe_intercept_jax)

    def actions(state, seat, which):
        aim._first_engine_hit_turn_jax, aim._search_safe_intercept_jax = which
        feats = build_world_features_from_state(state, seat)
        out = compute_actions(feats, _modes_from_features(feats))
        jax.block_until_ready(out)
        return np.asarray(out)

    jact = jax.jit(lambda f, m: compute_actions(f, m))

    def act(s, seat):
        f = build_world_features_from_state(s, seat)
        return jact(f, _modes_from_features(f))

    checked = mism = 0
    try:
        for seed in range(4):
            st = reset(seed=seed, num_agents=2)
            for turn in range(70):
                ea = (
                    jnp.full(
                        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, jnp.float32
                    )
                    .at[0]
                    .set(act(st, 0))
                    .at[1]
                    .set(act(st, 1))
                )
                st, _, term = jax.jit(env_step)(st, ea)
                if bool(term):
                    break
                if turn in (5, 25, 50, 68):
                    for seat in (0, 1):
                        new = actions(st, seat, new_fns)
                        old = actions(st, seat, old_fns)
                        ok = (
                            np.array_equal(new[:, 0], old[:, 0])
                            and np.array_equal(new[:, 2], old[:, 2])
                            and np.allclose(new[:, 1], old[:, 1], atol=1e-3)
                        )
                        if not ok:
                            mism += 1
                            print(f"MISMATCH seed={seed} turn={turn} seat={seat}")
                        checked += 1
    finally:
        aim._first_engine_hit_turn_jax, aim._search_safe_intercept_jax = new_fns

    verdict = "PASS" if mism == 0 else "FAIL"
    print(f"engine-vectorize identity: checked={checked} mism={mism} {verdict}")


if __name__ == "__main__":
    main()
