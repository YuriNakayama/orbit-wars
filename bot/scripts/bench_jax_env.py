"""Quick benchmark for the JAX env vs vendor sim.

Usage::

    uv run python scripts/bench_jax_env.py vendor --episodes 16 --seed 0
    uv run python scripts/bench_jax_env.py jax --episodes 16 --vmap 1
    uv run python scripts/bench_jax_env.py jax --episodes 16 --vmap 16
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import typer

# Vendor sim path.
_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "simulator" / "python"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_vendor.orbit_wars import interpreter as vendor_interpreter  # noqa: E402

from jax_env.reset import reset  # noqa: E402
from jax_env.state import EnvState  # noqa: E402
from jax_env.step import empty_actions, step  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = logging.getLogger(__name__)


def _run_vendor_episode(seed: int, max_steps: int = 500) -> int:
    """Run one vendor episode with no-op actions; return final step."""
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(), action=None, status="ACTIVE", reward=0
        )
        for _ in range(2)
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(episodeSteps=500, shipSpeed=6.0, cometSpeed=4.0),
        info={"seed": seed},
        done=False,
    )
    vendor_interpreter(state, env)
    for _ in range(max_steps):
        if state[0].status == "DONE":
            break
        for s in state:
            s.action = []
        vendor_interpreter(state, env)
    return int(getattr(state[0].observation, "step", 0))


@app.command("vendor")
def bench_vendor(
    episodes: int = typer.Option(16, help="Number of episodes"),
    seed: int = typer.Option(0, help="Base seed"),
    max_steps: int = typer.Option(500, help="Max steps per episode"),
) -> None:
    """Benchmark vendor `interpreter` end-to-end."""
    t0 = time.perf_counter()
    for i in range(episodes):
        _run_vendor_episode(seed + i, max_steps=max_steps)
    elapsed = time.perf_counter() - t0
    typer.echo(
        f"vendor: episodes={episodes} wall={elapsed:.2f}s "
        f"per_ep={elapsed / episodes:.3f}s"
    )


def _run_jax_episode_jit(
    state_init: EnvState, max_steps: int
) -> Callable[[EnvState], tuple[EnvState, jax.Array]]:
    """Closure factory for a jit-compiled single-episode rollout."""
    actions = empty_actions()

    def body(state: EnvState, _: jax.Array) -> tuple[EnvState, jax.Array]:
        new_state, _, term = step(state, actions)
        return new_state, term

    def run(state: EnvState) -> tuple[EnvState, jax.Array]:
        final, terms = jax.lax.scan(body, state, jnp.arange(max_steps))
        return final, terms

    return jax.jit(run)


@app.command("jax")
def bench_jax(
    episodes: int = typer.Option(16, help="Number of episodes"),
    device: str = typer.Option("cpu", help="JAX device: cpu | gpu"),
    vmap: int = typer.Option(1, help="Batch size for vmap"),
    seed: int = typer.Option(0, help="Base seed"),
    max_steps: int = typer.Option(500, help="Max steps per episode"),
) -> None:
    """Benchmark JAX env. Uses `lax.scan` for the inner loop."""
    if device != jax.default_backend():
        typer.echo(f"requested device={device} default={jax.default_backend()}")

    # Build initial states (CPU). vmap across episodes if requested.
    states = [reset(seed + i, num_agents=2) for i in range(episodes)]

    # Compile single-episode rollout.
    runner = _run_jax_episode_jit(states[0], max_steps)

    # Warm-up (compile cost not counted).
    final, _ = runner(states[0])
    final.step.block_until_ready()

    if vmap <= 1:
        t0 = time.perf_counter()
        for s in states:
            final, _ = runner(s)
            final.step.block_until_ready()
        elapsed = time.perf_counter() - t0
        typer.echo(
            f"jax (vmap=1): episodes={episodes} wall={elapsed:.2f}s "
            f"per_ep={elapsed / episodes:.3f}s"
        )
        return

    # Batched via vmap. Stack the per-episode states into a batched pytree.
    if episodes % vmap != 0:
        typer.echo(
            f"trim episodes {episodes} -> {(episodes // vmap) * vmap} for vmap fit"
        )
        episodes = (episodes // vmap) * vmap
        states = states[:episodes]

    batched_runner = jax.jit(jax.vmap(lambda s: runner(s)))
    # Warmup the vmap variant.
    batch0 = jax.tree.map(lambda *xs: jnp.stack(xs), *states[:vmap])
    final, _ = batched_runner(batch0)
    final.step.block_until_ready()

    t0 = time.perf_counter()
    for b in range(0, episodes, vmap):
        batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states[b : b + vmap])
        final, _ = batched_runner(batch)
        final.step.block_until_ready()
    elapsed = time.perf_counter() - t0
    typer.echo(
        f"jax (vmap={vmap}): episodes={episodes} wall={elapsed:.2f}s "
        f"per_ep={elapsed / episodes:.3f}s"
    )


if __name__ == "__main__":
    app()
