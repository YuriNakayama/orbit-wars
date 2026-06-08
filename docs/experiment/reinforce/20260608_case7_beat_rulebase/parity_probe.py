"""段2 parity probe: does the JAX training opponent (baseline_jax_full) produce
the same actions as the real Python baseline_v1 on identical board states?

Compares compute_actions_jax(state, seat) [JAX, used in H4 training] vs
_host_python_v1_action(state, seat) [real v1, used at eval] over many random
mid-game states. High disagreement => train/eval opponent parity gap explains
why H4 learned to beat the JAX opponent but not the real one.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case1.baseline_jax_full import compute_actions_jax
from pipeline.reinforce.case7.training.rollout_jax import _host_python_v1_action


def _norm(arr) -> set[tuple]:
    """Normalize an (L,3) action array to a set of (from_pid, angle_bucket, ships)
    launches, dropping -1 sentinel rows. Angle bucketed to 1 degree to ignore
    float noise."""
    a = np.asarray(arr)
    out = set()
    for row in a:
        pid = int(round(float(row[0])))
        if pid < 0:
            continue
        ang = int(round(float(row[1])))  # already degrees-ish
        ships = int(round(float(row[2])))
        out.add((pid, ang, ships))
    return out


def main() -> None:
    n_states = 40
    seat = 1
    exact = 0
    jaccard_sum = 0.0
    both_empty = 0
    jax_fires = 0
    py_fires = 0
    details = []
    for s in range(n_states):
        state = reset(seed=s, num_agents=2)
        # advance a random number of steps with noop to reach a mid-game state
        n_adv = (s % 12) * 3
        noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
        for _ in range(n_adv):
            state, _, term = step(state, noop)
            if bool(term):
                break
        jax_act = _norm(compute_actions_jax(state, seat))
        py_act = _norm(_host_python_v1_action(state, seat))
        if jax_act:
            jax_fires += 1
        if py_act:
            py_fires += 1
        if not jax_act and not py_act:
            both_empty += 1
            exact += 1
            jaccard_sum += 1.0
            continue
        inter = len(jax_act & py_act)
        union = len(jax_act | py_act)
        j = inter / union if union else 1.0
        jaccard_sum += j
        if jax_act == py_act:
            exact += 1
        if s < 8:
            details.append(
                f"  s{s} adv{n_adv}: jax={sorted(jax_act)} py={sorted(py_act)} J={j:.2f}"
            )

    print(f"states={n_states} seat={seat}")
    print(f"exact_match={exact}/{n_states} ({100*exact/n_states:.0f}%)")
    print(f"mean_jaccard={jaccard_sum/n_states:.3f}")
    print(f"both_empty(noop)={both_empty}  jax_fires={jax_fires}  py_fires={py_fires}")
    print("samples:")
    print("\n".join(details))


if __name__ == "__main__":
    main()
