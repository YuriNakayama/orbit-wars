"""End-to-end JAX PPO training driver for reinforce/case1.

Glues `collect_rollout_jax` (W4-a) + `ppo_update_jax` (W4-c) into a
PPO learning loop. Targets the same config schema as `train.py`
(YAML at `pipeline/reinforce/case1/configs/*.yaml`) so the existing
runpod_io case wiring (train_module pointer) works unchanged.

Pipeline per iteration:
  1) `collect_rollout_jax(model, key, episodes_per_iter, horizon=500)`
     → JaxRolloutBatch (B, T, ...) of JAX arrays.
  2) Compute GAE on the (B, T) reward/value tensors; flatten to
     (N=B*T, ...) FlatRollout.
  3) `ppo_update_jax(model, bc_ref, optimizer, opt_state, flat, cfg)`
     → new model + opt_state + stats.

Saves the latest model as `best.pt` (via numpy-side conversion so the
PyTorch reinforce/case1 inference path can still load it via the same
`load_bc_weights` adapter) and writes `metrics.json` per iter.

Note on the BC warm-start: we load the PyTorch BC weights into the JAX
model at startup (using `load_bc_weights_jax`). Optionally we freeze a
copy as `bc_reference` for the KL anchor — set via `bc_warmstart` in
the YAML, same shape as `train.py`.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import typer
import yaml

from pipeline.reinforce.case1.policy.featurizer_jax import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.reinforce.case1.policy.model_jax import (
    ActorCriticJax,
    ModelConfigJax,
    load_bc_weights_jax,
)
from pipeline.reinforce.case1.training.ppo_jax import (
    FlatRollout,
    PPOConfigJax,
    make_optimizer,
    ppo_update_jax,
)
from pipeline.reinforce.case1.training.rollout_jax import (
    JaxRolloutBatch,
    collect_rollout_jax,
)
from utils.repo_root import absolute_under_repo

# W6-a: wrap the entire epoch×minibatch loop in a single jit so the
# update is compiled once per (model_pytree_shape, rollout_shape, cfg)
# tuple. Each subsequent call reuses the compiled XLA executable —
# critical for keeping PPO updates on-device on RunPod GPUs.
_ppo_update_jit = eqx.filter_jit(ppo_update_jax)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainReport:
    iterations_run: int
    total_episodes: int
    best_win_rate: float
    runtime_seconds: float


def _resolve_run_dir(cfg: dict[str, Any]) -> Path:
    env_run_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    base = Path(absolute_under_repo(cfg["output"]["run_dir"], start=Path(__file__)))
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"local_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compute_gae(
    rewards: jax.Array,  # (B, T)
    values: jax.Array,  # (B, T)
    done_mask: jax.Array,  # (B, T) bool — True = valid step
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE-λ advantage + returns per (B, T). Done_mask=False steps zeroed."""
    B, T = rewards.shape
    next_adv = jnp.zeros((B,), dtype=jnp.float32)
    next_val = jnp.zeros((B,), dtype=jnp.float32)

    def step(
        carry: tuple[jax.Array, jax.Array], idx: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        next_adv, next_val = carry
        t = T - 1 - idx
        r = rewards[:, t]
        v = values[:, t]
        valid = done_mask[:, t].astype(jnp.float32)
        delta = r + gamma * next_val - v
        adv = delta + gamma * lam * next_adv
        # Zero out invalid steps so they don't contribute to the bootstrap.
        adv = adv * valid
        return (adv, v * valid + next_val * (1 - valid)), adv

    _, adv_seq = jax.lax.scan(step, (next_adv, next_val), jnp.arange(T))
    # adv_seq has shape (T, B); reverse along time, transpose to (B, T).
    adv_seq = adv_seq[::-1].T
    returns = adv_seq + values
    return adv_seq, returns


def _flatten_rollout(
    rollout: JaxRolloutBatch, gamma: float, gae_lambda: float
) -> FlatRollout:
    """Convert (B, T, ...) rollout to (N, ...) flat with GAE."""
    advantages, returns = _compute_gae(
        rollout.rewards, rollout.values, rollout.done_mask, gamma, gae_lambda
    )

    def flat(x: jax.Array) -> jax.Array:
        # (B, T, ...) -> (B*T, ...) — but only keep valid steps.
        return x.reshape((-1,) + x.shape[2:])

    return FlatRollout(
        planet_feats=flat(rollout.planet_feats),
        planet_mask=flat(rollout.planet_mask),
        my_planet_mask=flat(rollout.my_planet_mask),
        target_mask=flat(rollout.target_mask),
        global_feats=flat(rollout.global_feats),
        template_ctx=flat(rollout.template_ctx),
        candidate_feats=flat(rollout.candidate_feats),
        candidate_mask=flat(rollout.candidate_mask),
        candidate_pid=flat(rollout.candidate_pid),
        target_slot=flat(rollout.target_slot),
        log1p_ships=flat(rollout.log1p_ships),
        log_probs=flat(rollout.log_probs),
        values=flat(rollout.values),
        advantages=flat(advantages),
        returns=flat(returns),
        done_mask=flat(rollout.done_mask),
    )


def _save_best_pt(model: ActorCriticJax, path: Path) -> None:
    """Dump the model leaves into a numpy archive at `path`.

    The PyTorch inference path uses `torch.load(.pt, weights_only=True)`
    expecting a state_dict. For W5 smoke we emit the JAX leaves as a
    numpy `.npz` and rename to `.pt` (since the runpod artifacts
    uploader is path-string based). Inference re-loading is a follow-
    up — out of scope for the bench wall-clock measurement.
    """
    leaves = eqx.filter(model, eqx.is_array)
    flat_leaves = jax.tree.leaves(leaves)
    # Just store as a dict of numpy arrays keyed by index.
    arrays = {f"leaf_{i}": np.asarray(arr) for i, arr in enumerate(flat_leaves)}
    np.savez(str(path), **arrays)  # type: ignore[arg-type]


def _build_optimizer_state(model: ActorCriticJax, cfg: PPOConfigJax) -> tuple[Any, Any]:
    """Returns (optimizer, opt_state)."""
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state


def _build_model(cfg_dict: dict[str, Any]) -> ActorCriticJax:
    m = cfg_dict.get("model", {})
    model_cfg = ModelConfigJax(
        planet_in_dim=PLANET_FEAT_DIM,
        global_in_dim=GLOBAL_FEAT_DIM,
        hidden=int(m.get("hidden", 192)),
        attn_heads=int(m.get("attn_heads", 8)),
        inducing_points=int(m.get("inducing_points", 24)),
        encoder_layers=int(m.get("encoder_layers", 4)),
        head_dropout=float(m.get("head_dropout", 0.0)),
        ship_log_std_init=float(m.get("ship_log_std_init", 0.5)),
        no_op_bias=float(m.get("no_op_bias", 8.0)),
    )
    return ActorCriticJax.from_init(jax.random.PRNGKey(0), model_cfg)


def _maybe_load_bc(model: ActorCriticJax, cfg_dict: dict[str, Any]) -> ActorCriticJax:
    bc_cfg = cfg_dict.get("bc_warmstart", {})
    if not bc_cfg.get("enabled", False):
        return model
    weights_path = absolute_under_repo(bc_cfg["weights_path"], start=Path(__file__))
    if not Path(weights_path).exists():
        logger.warning("BC weights not found at %s — skipping", weights_path)
        return model
    new_model, loaded, missing = load_bc_weights_jax(model, str(weights_path))
    logger.info("BC warm-start loaded=%d missing=%d", loaded, missing)
    return new_model


def _build_ppo_cfg(cfg_dict: dict[str, Any]) -> PPOConfigJax:
    t = cfg_dict.get("training", {})
    return PPOConfigJax(
        clip_eps=float(t.get("clip_eps", 0.2)),
        value_coef=float(t.get("value_coef", 0.5)),
        entropy_coef=float(t.get("entropy_coef", 0.003)),
        kl_beta=float(t.get("kl_beta", 0.1)),
        epochs=int(t.get("ppo_epochs", 4)),
        minibatch_size=int(t.get("minibatch_size", 64)),
        max_grad_norm=float(t.get("max_grad_norm", 0.5)),
        normalize_advantage=True,
        target_kl=t.get("target_kl"),
        lr=float(t.get("lr", 1.0e-4)),
        weight_decay=float(t.get("weight_decay", 1.0e-5)),
        lr_end=float(t.get("lr_end", 0.0)),
        lr_schedule_steps=int(t.get("lr_schedule_steps", 0)),
    )


def _run_iter(
    model: ActorCriticJax,
    bc_reference: ActorCriticJax | None,
    optimizer: Any,
    opt_state: Any,
    cfg: PPOConfigJax,
    key: jax.Array,
    iter_idx: int,
    episodes_per_iter: int,
    horizon: int,
    shaping_coef: float,
    gamma: float,
    gae_lambda: float,
    base_seed: int,
    opponent: str = "noop",
    shaping_mode: str = "ships",
) -> tuple[ActorCriticJax, Any, dict[str, Any]]:
    rollout_key, update_key = jax.random.split(key)

    t0 = time.perf_counter()
    rollout = collect_rollout_jax(
        model,
        rollout_key,
        episodes_per_iter=episodes_per_iter,
        horizon=horizon,
        shaping_coef=shaping_coef,
        seat=0,
        seed=base_seed + iter_idx * 10_000,
        opponent=opponent,
        shaping_mode=shaping_mode,
    )
    rollout.planet_feats.block_until_ready()
    rollout_secs = time.perf_counter() - t0

    flat = _flatten_rollout(rollout, gamma, gae_lambda)

    t0 = time.perf_counter()
    model, opt_state, stats = _ppo_update_jit(
        model, bc_reference, optimizer, opt_state, flat, cfg, update_key
    )
    # Block on a small leaf to ensure compute is materialized.
    _ = float(stats.policy_loss)
    update_secs = time.perf_counter() - t0

    outcomes = np.asarray(rollout.episode_outcomes)
    win_rate = float(np.mean(outcomes > 0))
    return (
        model,
        opt_state,
        {
            "iter": iter_idx,
            "rollout_secs": rollout_secs,
            "update_secs": update_secs,
            "win_rate": win_rate,
            "policy_loss": float(stats.policy_loss),
            "value_loss": float(stats.value_loss),
            "entropy": float(stats.entropy),
            "approx_kl": float(stats.approx_kl),
            "bc_kl": float(stats.bc_kl),
            "clip_fraction": float(stats.clip_fraction),
            "epochs_run": float(stats.epochs_run),
        },
    )


app = typer.Typer(add_completion=False)

_DEFAULT_CONFIG = typer.Option(
    Path("pipeline/reinforce/case1/configs/train_jax.yaml"),
    "--config",
    help="Path to PPO YAML config (relative to bot/).",
)


@app.command()
def main(config: Path = _DEFAULT_CONFIG) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # config is relative to bot/ (matches train.py contract). Resolve
    # relative to the repo `bot/` directory so callers can pass either
    # `pipeline/reinforce/case1/configs/foo.yaml` or an absolute path.
    if config.is_absolute():
        cfg_path = config
    else:
        cfg_path = Path(__file__).resolve().parents[4] / config
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    with cfg_path.open() as f:
        cfg_dict = yaml.safe_load(f)

    run_dir = _resolve_run_dir(cfg_dict)
    logger.info("run_dir=%s config=%s", run_dir, cfg_path)

    t_cfg = cfg_dict.get("training", {})
    iterations = int(t_cfg.get("iterations", 5))
    episodes_per_iter = int(t_cfg.get("episodes_per_iter", 16))
    horizon = int(t_cfg.get("horizon", 500))
    shaping_coef = float(t_cfg.get("shaping_coef", 0.001))
    gamma = float(t_cfg.get("gamma", 0.99))
    gae_lambda = float(t_cfg.get("gae_lambda", 0.95))
    seed = int(t_cfg.get("seed", 0))
    opponent = str(t_cfg.get("opponent", "noop"))
    shaping_mode = str(t_cfg.get("shaping_mode", "ships"))

    # Opponent curriculum (B2): use `early` opponent for iters < switch_iter,
    # `late` opponent afterwards. When `opponent` is not "curriculum" the
    # curriculum block is ignored and a single opponent runs for all iters.
    curriculum_cfg = t_cfg.get("opponent_curriculum", {}) or {}
    curriculum_switch_iter = int(curriculum_cfg.get("switch_iter", 0))
    curriculum_early = str(curriculum_cfg.get("early", "noop"))
    curriculum_late = str(curriculum_cfg.get("late", "baseline_jax_lite"))

    def _opponent_for_iter(it: int) -> str:
        if opponent != "curriculum":
            return opponent
        return curriculum_early if it < curriculum_switch_iter else curriculum_late

    ppo_cfg = _build_ppo_cfg(cfg_dict)
    model = _build_model(cfg_dict)
    model = _maybe_load_bc(model, cfg_dict)
    bc_reference: ActorCriticJax | None = (
        eqx.tree_at(lambda m: m, model, replace_fn=lambda x: x)
        if cfg_dict.get("bc_warmstart", {}).get("enabled", False)
        and ppo_cfg.kl_beta > 0
        else None
    )
    if bc_reference is not None:
        logger.info("BC reference frozen for KL anchor (kl_beta=%s)", ppo_cfg.kl_beta)

    optimizer, opt_state = _build_optimizer_state(model, ppo_cfg)

    if opponent == "curriculum":
        logger.info(
            "starting JAX PPO loop: iters=%d episodes_per_iter=%d horizon=%d "
            "opponent=curriculum (early=%s, late=%s, switch_iter=%d)",
            iterations,
            episodes_per_iter,
            horizon,
            curriculum_early,
            curriculum_late,
            curriculum_switch_iter,
        )
    else:
        logger.info(
            "starting JAX PPO loop: iters=%d episodes_per_iter=%d horizon=%d "
            "opponent=%s",
            iterations,
            episodes_per_iter,
            horizon,
            opponent,
        )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    key = jax.random.PRNGKey(seed)
    best_win = 0.0
    for it in range(iterations):
        iter_opponent = _opponent_for_iter(it)
        key, k_iter = jax.random.split(key)
        model, opt_state, row = _run_iter(
            model,
            bc_reference,
            optimizer,
            opt_state,
            ppo_cfg,
            k_iter,
            it,
            episodes_per_iter,
            horizon,
            shaping_coef,
            gamma,
            gae_lambda,
            seed,
            opponent=iter_opponent,
            shaping_mode=shaping_mode,
        )
        row["opponent"] = iter_opponent
        row["shaping_mode"] = shaping_mode
        history.append(row)
        logger.info(
            (
                "iter=%d rollout=%.2fs update=%.2fs win=%.3f loss=%.4f "
                "approx_kl=%.4f bc_kl=%.4f"
            ),
            row["iter"],
            row["rollout_secs"],
            row["update_secs"],
            row["win_rate"],
            row["policy_loss"],
            row["approx_kl"],
            row["bc_kl"],
        )
        best_win = max(best_win, row["win_rate"])
        # Save model snapshot.
        _save_best_pt(model, run_dir / "best.pt")

    runtime = time.perf_counter() - started
    report = TrainReport(
        iterations_run=iterations,
        total_episodes=iterations * episodes_per_iter,
        best_win_rate=best_win,
        runtime_seconds=runtime,
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                **asdict(report),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
                "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                "history": history,
            },
            indent=2,
        )
    )
    logger.info(
        "training done iters=%d runtime=%.1fs best_win=%.3f",
        iterations,
        runtime,
        best_win,
    )


if __name__ == "__main__":
    app()
