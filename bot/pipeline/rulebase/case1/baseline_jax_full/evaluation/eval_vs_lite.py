"""Self-play eval: baseline_jax_full vs baseline_jax_lite via jax_env.

Goal: quantify how much stronger `_full` is than `_lite` in fixed seeds.
Used to bound expected case2 PPO training opponent boost. Not a substitute
for baseline_v1 evaluation (which requires kaggle_environments).
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import typer

from jax_env.constants import MAX_PLANETS, NUM_AGENTS_MAX
from jax_env.reset import reset
from jax_env.step import MAX_LAUNCHES_PER_AGENT, step as jax_env_step
from pipeline.rulebase.case1.baseline_jax import (
    compute_actions_jax as lite_actions,
)
from pipeline.rulebase.case1.baseline_jax_full import (
    compute_actions_jax as full_actions,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="baseline_jax_full vs baseline_jax_lite eval.")


def _play_one(seed: int, swap: bool, horizon: int = 500) -> tuple[int, int]:
    """Return (full_score, lite_score) after one episode.

    score = ships of the agent's owned planets at terminal step
            (proxy for win — vendor terminal reward depends on planet ownership).
    swap: if True, seat assignment is full=seat1 / lite=seat0; else reversed.
    """
    state = reset(seed)
    full_seat = 1 if swap else 0
    lite_seat = 0 if swap else 1

    @jax.jit
    def step_loop(state):
        def body(carry, _):
            state, done = carry
            empty_actions = jnp.full(
                (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3),
                -1.0, dtype=jnp.float32,
            )
            full_row = full_actions(state, full_seat)
            lite_row = lite_actions(state, lite_seat)
            env_actions = empty_actions.at[full_seat].set(full_row)
            env_actions = env_actions.at[lite_seat].set(lite_row)
            new_state, _r, term = jax_env_step(state, env_actions)
            done_now = done | (term.astype(jnp.bool_))
            kept = jax.tree.map(
                lambda new, old: jnp.where(done, old, new), new_state, state
            )
            return (kept, done_now), None

        (final, _), _ = jax.lax.scan(
            body, (state, jnp.bool_(False)), jnp.arange(horizon)
        )
        return final

    final = step_loop(state)
    owner = final.planet_owner
    ships = final.planet_ships
    valid = final.planet_valid
    full_score = int(
        jnp.sum(jnp.where(valid & (owner == full_seat), ships, 0))
    )
    lite_score = int(
        jnp.sum(jnp.where(valid & (owner == lite_seat), ships, 0))
    )
    return full_score, lite_score


@app.command()
def main(
    episodes: int = typer.Option(20, "--episodes", "-n"),
    horizon: int = typer.Option(500, "--horizon"),
    base_seed: int = typer.Option(42, "--base-seed"),
):
    """Run N episodes, half with seat swap, return win rates."""
    full_wins = 0
    lite_wins = 0
    draws = 0
    score_diff_sum = 0
    for i in range(episodes):
        swap = bool(i % 2)
        seed = base_seed + i
        f, l = _play_one(seed, swap=swap, horizon=horizon)
        if f > l:
            full_wins += 1
            outcome = "FULL"
        elif l > f:
            lite_wins += 1
            outcome = "LITE"
        else:
            draws += 1
            outcome = "DRAW"
        score_diff_sum += f - l
        print(
            f"  ep{i + 1:>3d} seed={seed} swap={int(swap)} "
            f"full_score={f:>4d} lite_score={l:>4d} -> {outcome}"
        )

    total = episodes - draws
    full_rate = full_wins / total if total > 0 else 0.0
    print()
    print(f"=== baseline_jax_full vs baseline_jax_lite ({episodes} eps, horizon={horizon}) ===")
    print(f"FULL wins: {full_wins}/{episodes} = {full_wins / episodes:.3f}")
    print(f"LITE wins: {lite_wins}/{episodes} = {lite_wins / episodes:.3f}")
    print(f"draws: {draws}/{episodes}")
    print(f"win-rate (excl draws): {full_rate:.3f}")
    print(f"avg score diff (full - lite): {score_diff_sum / episodes:+.1f}")


if __name__ == "__main__":
    app()
