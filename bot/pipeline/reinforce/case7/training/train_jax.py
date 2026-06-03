"""End-to-end JAX PPO training driver for reinforce/case7.

Glues `collect_rollout_jax` (W4-a) + `ppo_update_jax` (W4-c) into a
PPO learning loop. Targets the same config schema as `train.py`
(YAML at `pipeline/reinforce/case7/configs/*.yaml`) so the existing
runpod_io case wiring (train_module pointer) works unchanged.

Pipeline per iteration:
  1) `collect_rollout_jax(model, key, episodes_per_iter, horizon=500)`
     → JaxRolloutBatch (B, T, ...) of JAX arrays.
  2) Compute GAE on the (B, T) reward/value tensors; flatten to
     (N=B*T, ...) FlatRollout.
  3) `ppo_update_jax(model, bc_ref, optimizer, opt_state, flat, cfg)`
     → new model + opt_state + stats.

Saves the latest model as `best.pt` (via numpy-side conversion so the
PyTorch reinforce/case7 inference path can still load it via the same
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

from pipeline.reinforce.case7.policy.featurizer_jax import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.reinforce.case7.policy.model_jax import (
    ActorCriticJax,
    ModelConfigJax,
    load_bc_weights_jax,
)
from pipeline.reinforce.case7.training.ppo_jax import (
    FlatRollout,
    PPOConfigJax,
    make_optimizer,
    ppo_update_jax,
)
from pipeline.reinforce.case7.training.rollout_jax import (
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
    """Dump the model leaves into a numpy npz archive written at exactly `path`.

    The leaves are stored as an npz payload but the file keeps the caller's
    name verbatim (e.g. `best.pt`). We must write through an open file handle:
    `np.savez(str(path))` appends a `.npz` suffix when given a path string, so
    `np.savez("best.pt")` would silently create `best.pt.npz` — leaving the
    onstart artifact uploaders (which look for `best.pt`) with nothing to ship.
    PyTorch inference re-loading of these JAX leaves is a follow-up.
    """
    leaves = eqx.filter(model, eqx.is_array)
    flat_leaves = jax.tree.leaves(leaves)
    # Just store as a dict of numpy arrays keyed by index.
    arrays = {f"leaf_{i}": np.asarray(arr) for i, arr in enumerate(flat_leaves)}
    with path.open("wb") as fh:
        np.savez(fh, **arrays)  # type: ignore[arg-type]


def _load_npz_into_model(model: ActorCriticJax, path: Path) -> ActorCriticJax:
    """Inverse of `_save_best_pt`: load `leaf_{i}` arrays from a best.pt npz and
    graft them back onto `model`'s array leaves (in tree order), returning the
    restored model. `model` supplies the static structure + non-array leaves, so
    it must be built from the same config as the checkpoint. Used for resume
    (continue training from a saved best.pt).
    """
    params, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree.flatten(params)
    with np.load(path) as data:
        if len(data.files) != len(flat):
            raise ValueError(
                f"resume checkpoint {path} has {len(data.files)} leaves but the "
                f"model built from config has {len(flat)}; config mismatch"
            )
        restored = [
            jnp.asarray(data[f"leaf_{i}"]).reshape(leaf.shape).astype(leaf.dtype)
            for i, leaf in enumerate(flat)
        ]
    new_params = jax.tree.unflatten(treedef, restored)
    restored_model: ActorCriticJax = eqx.combine(new_params, static)
    return restored_model


def _maybe_resume(model: ActorCriticJax, cfg_dict: dict[str, Any]) -> ActorCriticJax:
    """Resume from a previously saved best.pt (npz) when `training.resume_from`
    is set. Takes precedence over BC warm-start (a resume already carries the
    learned weights). Path is resolved under the repo root.
    """
    resume_from = cfg_dict.get("training", {}).get("resume_from")
    if not resume_from:
        return model
    path = absolute_under_repo(str(resume_from), start=Path(__file__))
    if not Path(path).exists():
        raise FileNotFoundError(f"training.resume_from not found: {path}")
    new_model = _load_npz_into_model(model, Path(path))
    logger.info("resumed weights from %s", path)
    return new_model


def _write_metrics(
    path: Path,
    *,
    iterations_run: int,
    total_episodes: int,
    best_win_rate: float,
    runtime_seconds: float,
    history: list[dict[str, Any]],
) -> None:
    """Write metrics.json. Called after EVERY iter so the reward/stat trace and
    opponent-pool distribution are readable mid-training (paired with the
    per-iter best.pt). `history` carries per-iter mean_reward / reward
    std-min-max / opponent / win_rate / value_loss / entropy / clip_fraction.
    """
    report = TrainReport(
        iterations_run=iterations_run,
        total_episodes=total_episodes,
        best_win_rate=best_win_rate,
        runtime_seconds=runtime_seconds,
    )
    path.write_text(
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


def _upload_best_to_s3(best_pt: Path, iteration: int, win_rate: float) -> None:
    """Stream the latest best.pt to S3 the moment a new best fires.

    Mirrors the imitation/case11 layer-1 safeguard: the onstart heartbeat
    (60s) and post-train upload still run, but RL iters can be minutes long,
    so a host preempt between best-update and the next heartbeat would lose
    the freshest weights. Uploading on every improvement closes that window.

    `ORBIT_WARS_BEST_S3_PREFIX` is set by onstart to the per-run artifacts
    directory (no trailing filename). We upload two objects:
      <prefix>/best_i{iter}_win{rate:.4f}.pt  — per-iter history (preempt-safe)
      <prefix>/best.pt                          — latest-best alias
    Outside RunPod the var is empty and we skip silently.
    """
    prefix = os.environ.get("ORBIT_WARS_BEST_S3_PREFIX", "").strip()
    if not (prefix and prefix.startswith("s3://")):
        return
    bucket, _, key_prefix = prefix[len("s3://") :].rstrip("/").partition("/")
    history_key = f"{key_prefix}/best_i{iteration}_win{win_rate:.4f}.pt"
    latest_key = f"{key_prefix}/best.pt"
    try:
        import boto3  # noqa: PLC0415

        s3 = boto3.client("s3")
        s3.upload_file(str(best_pt), bucket, history_key)
        s3.upload_file(str(best_pt), bucket, latest_key)
        logger.info(
            json.dumps(
                {
                    "event": "best_s3_upload",
                    "iter": iteration,
                    "history_uri": f"s3://{bucket}/{history_key}",
                    "latest_uri": f"s3://{bucket}/{latest_key}",
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "best.pt S3 upload failed iter=%d prefix=%s err=%s",
            iteration,
            prefix,
            exc,
        )


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
    coef_ship: float = 0.0,
    coef_planet: float = 0.0,
    shaping_clip: float = 0.0,
    dense_coef_ship: float = 0.0,
    dense_coef_planet: float = 0.0,
    time_bonus_coef: float = 0.0,
    time_penalty_coef: float = 0.0,
    opp_model: ActorCriticJax | None = None,
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
        coef_ship=coef_ship,
        coef_planet=coef_planet,
        shaping_clip=shaping_clip,
        dense_coef_ship=dense_coef_ship,
        dense_coef_planet=dense_coef_planet,
        time_bonus_coef=time_bonus_coef,
        time_penalty_coef=time_penalty_coef,
        opp_model=opp_model,
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
            "mean_reward": float(np.mean(outcomes)),
            "reward_std": float(np.std(outcomes)),
            "reward_min": float(np.min(outcomes)),
            "reward_max": float(np.max(outcomes)),
            "policy_loss": float(stats.policy_loss),
            "value_loss": float(stats.value_loss),
            "entropy": float(stats.entropy),
            "approx_kl": float(stats.approx_kl),
            "bc_kl": float(stats.bc_kl),
            "clip_fraction": float(stats.clip_fraction),
            "epochs_run": float(stats.epochs_run),
        },
    )


class _OpponentPool:
    """H2 (PFSP): a host-side FIFO pool of frozen self snapshots.

    Holds up to `cap` Equinox models (immutable pytrees). `push` adds the
    latest snapshot (FIFO drop when full); `sample` returns one uniformly.
    Selection happens host-side (Python), so the snapshot fed to
    `collect_rollout_jax(opp_model=...)` is just a weights swap into a
    structurally-identical pytree — the scan/vmap trace stays single.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(1, cap)
        self._models: list[ActorCriticJax] = []

    def push(self, model: ActorCriticJax) -> None:
        kept = self._models[-(self._cap - 1) :] if self._cap > 1 else []
        self._models = [*kept, model]

    def sample(self, rng: np.random.Generator) -> ActorCriticJax:
        return self._models[int(rng.integers(len(self._models)))]

    def models(self) -> list[ActorCriticJax]:
        return list(self._models)

    def __len__(self) -> int:
        return len(self._models)


@dataclass
class _OpponentEntry:
    """One selectable opponent: a rule mode name + optional self-snapshot model."""

    opponent: str  # "baseline_jax_full" or "self_snapshot"
    model: ActorCriticJax | None  # None for rule opponents
    win_ema: float  # current agent's win-rate EMA vs this entry (x in f_hard)


class _PrioritizedOpponentSelector:
    """H4 (PFSP): pick opponents by f_hard(x) = (1 - x)^p.

    Entries = [baseline_jax_full] + the current snapshot pool. `x` is the
    learner's EMA win-rate vs each entry (init 0.5 = unknown). Harder
    opponents (low win-rate) get higher selection weight, concentrating
    learning on what the agent cannot yet beat (AlphaStar main-agent recipe).
    Selection is host-side; only the chosen (opponent, model) is threaded into
    `collect_rollout_jax`, so the scan/vmap trace stays single.
    """

    def __init__(self, p: float, ema: float, init_win: float = 0.5) -> None:
        self._p = p
        self._ema = ema
        self._init_win = init_win
        self._entries: list[_OpponentEntry] = []

    def set_entries(
        self, pool_models: list[ActorCriticJax], include_full: bool
    ) -> None:
        """Rebuild entries from the current pool (+ baseline_jax_full).

        Preserves the win_ema of carried-over snapshot entries by index from
        the tail (FIFO pool), and seeds new entries at init_win.
        """
        prev_snaps = [e for e in self._entries if e.opponent == "self_snapshot"]
        new_entries: list[_OpponentEntry] = []
        if include_full:
            full_prev = next(
                (e for e in self._entries if e.opponent == "baseline_jax_full"), None
            )
            new_entries.append(
                _OpponentEntry(
                    "baseline_jax_full",
                    None,
                    full_prev.win_ema if full_prev else self._init_win,
                )
            )
        # Align pool models to the tail of prev snapshot EMAs (FIFO order).
        carried = prev_snaps[-len(pool_models) :] if pool_models else []
        pad = len(pool_models) - len(carried)
        for i, m in enumerate(pool_models):
            ema = carried[i - pad].win_ema if i >= pad else self._init_win
            new_entries.append(_OpponentEntry("self_snapshot", m, ema))
        self._entries = new_entries

    def sample(self, rng: np.random.Generator) -> tuple[int, _OpponentEntry]:
        weights = np.array([(1.0 - e.win_ema) ** self._p for e in self._entries])
        total = weights.sum()
        probs = (
            weights / total
            if total > 0
            else np.ones(len(self._entries)) / len(self._entries)
        )
        idx = int(rng.choice(len(self._entries), p=probs))
        return idx, self._entries[idx]

    def update(self, idx: int, win_rate: float) -> None:
        e = self._entries[idx]
        e.win_ema = self._ema * e.win_ema + (1.0 - self._ema) * win_rate

    def __len__(self) -> int:
        return len(self._entries)


app = typer.Typer(add_completion=False)

_DEFAULT_CONFIG = typer.Option(
    Path("pipeline/reinforce/case7/configs/train_jax.yaml"),
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
    # `pipeline/reinforce/case7/configs/foo.yaml` or an absolute path.
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
    coef_ship = float(t_cfg.get("coef_ship", 0.0))
    coef_planet = float(t_cfg.get("coef_planet", 0.0))
    shaping_clip = float(t_cfg.get("shaping_clip", 0.0))
    dense_coef_ship = float(t_cfg.get("dense_coef_ship", 0.0))
    dense_coef_planet = float(t_cfg.get("dense_coef_planet", 0.0))
    time_bonus_coef = float(t_cfg.get("time_bonus_coef", 0.0))
    time_penalty_coef = float(t_cfg.get("time_penalty_coef", 0.0))

    # Opponent curriculum (B2): use `early` opponent for iters < switch_iter,
    # `late` opponent afterwards. When `opponent` is not "curriculum" the
    # curriculum block is ignored and a single opponent runs for all iters.
    # H6b: optional 3-stage curriculum — when `mid` and `mid_switch_iter` are
    # set in the config, iter range [switch_iter, mid_switch_iter) uses `mid`
    # (e.g. baseline_jax_lite) and iter >= mid_switch_iter uses `late`. This
    # gives PPO a winnable intermediate opponent before facing the real
    # python_v1 (which collapsed to 0/8 with the 2-stage noop→python_v1 path).
    curriculum_cfg = t_cfg.get("opponent_curriculum", {}) or {}
    curriculum_switch_iter = int(curriculum_cfg.get("switch_iter", 0))
    curriculum_early = str(curriculum_cfg.get("early", "noop"))
    curriculum_mid = str(curriculum_cfg.get("mid", "")).strip() or None
    curriculum_mid_switch_iter = int(
        curriculum_cfg.get("mid_switch_iter", curriculum_switch_iter)
    )
    curriculum_late = str(curriculum_cfg.get("late", "baseline_jax_lite"))

    # H2 (PFSP pool): when late == "pool", maintain a FIFO snapshot pool and,
    # each late iter, play either a pooled snapshot or baseline_jax_full.
    pool_cfg = t_cfg.get("opponent_pool", {}) or {}
    pool_snapshot_every = int(pool_cfg.get("snapshot_every", 10))
    pool_cap = int(pool_cfg.get("cap", 5))
    pool_late_full_prob = float(pool_cfg.get("late_full_prob", 0.5))
    use_pool = opponent == "curriculum" and curriculum_late == "pool"
    # H4: when priority == "f_hard", pick the late opponent by (1-x)^p instead
    # of the uniform pool/full mix. priority == "uniform" reproduces H2.
    pool_priority = str(pool_cfg.get("priority", "uniform"))
    pool_priority_p = float(pool_cfg.get("priority_p", 2.0))
    pool_priority_ema = float(pool_cfg.get("priority_ema", 0.7))
    use_priority = use_pool and pool_priority == "f_hard"

    def _opponent_for_iter(it: int) -> str:
        if opponent != "curriculum":
            return opponent
        if it < curriculum_switch_iter:
            return curriculum_early
        # H6b 3-stage: iter range [switch_iter, mid_switch_iter) uses `mid`
        # (e.g. baseline_jax_lite as a winnable intermediate).
        if (
            curriculum_mid is not None
            and curriculum_switch_iter <= it < curriculum_mid_switch_iter
        ):
            return curriculum_mid
        # When late == "pool", the concrete opponent is decided in-loop
        # (pool snapshot vs baseline_jax_full); return the base curriculum_late.
        return curriculum_late

    ppo_cfg = _build_ppo_cfg(cfg_dict)
    model = _build_model(cfg_dict)
    model = _maybe_load_bc(model, cfg_dict)
    # Freeze the KL-anchor reference from the BC-loaded model BEFORE any resume,
    # so the anchor stays the BC policy (not the resumed checkpoint).
    bc_reference: ActorCriticJax | None = (
        eqx.tree_at(lambda m: m, model, replace_fn=lambda x: x)
        if cfg_dict.get("bc_warmstart", {}).get("enabled", False)
        and ppo_cfg.kl_beta > 0
        else None
    )
    if bc_reference is not None:
        logger.info("BC reference frozen for KL anchor (kl_beta=%s)", ppo_cfg.kl_beta)
    # Resume (continue training from a saved best.pt) overrides the live weights.
    model = _maybe_resume(model, cfg_dict)

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
    # H1 (PFSP foundation): freeze the start-of-training model as the initial
    # self-snapshot opponent. H2: also seed a FIFO snapshot pool with it so the
    # pool is non-empty before the first periodic push.
    opp_snapshot = model
    pool = _OpponentPool(pool_cap)
    pool.push(model)
    pool_rng = np.random.default_rng(seed)
    # H4: prioritized selector over [baseline_jax_full] + pool snapshots.
    selector = _PrioritizedOpponentSelector(pool_priority_p, pool_priority_ema)
    if use_priority:
        selector.set_entries(pool.models(), include_full=True)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    key = jax.random.PRNGKey(seed)
    best_win = 0.0
    for it in range(iterations):
        iter_opponent = _opponent_for_iter(it)
        iter_opp_model = opp_snapshot
        sel_idx = -1
        if use_pool and it >= curriculum_switch_iter:
            if use_priority:
                # H4: f_hard=(1-x)^p over full + pool snapshots.
                sel_idx, entry = selector.sample(pool_rng)
                iter_opponent = entry.opponent
                iter_opp_model = (
                    entry.model if entry.model is not None else opp_snapshot
                )
            elif pool_rng.random() < pool_late_full_prob:
                # H2: uniform full/pool mix.
                iter_opponent = "baseline_jax_full"
            else:
                iter_opponent = "self_snapshot"
                iter_opp_model = pool.sample(pool_rng)
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
            coef_ship=coef_ship,
            coef_planet=coef_planet,
            shaping_clip=shaping_clip,
            dense_coef_ship=dense_coef_ship,
            dense_coef_planet=dense_coef_planet,
            time_bonus_coef=time_bonus_coef,
            time_penalty_coef=time_penalty_coef,
            opp_model=iter_opp_model,
        )
        # H4: update the selected entry's win-rate EMA with this iter's outcome.
        if use_priority and sel_idx >= 0:
            selector.update(sel_idx, row["win_rate"])
        # H2/H4: refresh the pool with the just-updated model every K iters so
        # the opponent distribution tracks the learner instead of staying frozen.
        if use_pool and (it + 1) % pool_snapshot_every == 0:
            pool.push(model)
            if use_priority:
                selector.set_entries(pool.models(), include_full=True)
        row["opponent"] = iter_opponent
        # Pool state for this iter: how many past-self snapshots are live and
        # which one (if any) was sampled, so the opponent distribution can be
        # reconstructed mid-run. sel_idx >= 0 only under f_hard priority.
        row["pool_size"] = len(pool) if use_pool else 0
        row["pool_sel_idx"] = sel_idx
        row["shaping_mode"] = shaping_mode
        row["coef_ship"] = coef_ship
        row["coef_planet"] = coef_planet
        row["shaping_clip"] = shaping_clip
        row["dense_coef_ship"] = dense_coef_ship
        row["dense_coef_planet"] = dense_coef_planet
        row["time_bonus_coef"] = time_bonus_coef
        row["time_penalty_coef"] = time_penalty_coef
        history.append(row)
        logger.info(
            (
                "iter=%d opp=%s rollout=%.2fs update=%.2fs win=%.3f reward=%.4f "
                "loss=%.4f approx_kl=%.4f bc_kl=%.4f"
            ),
            row["iter"],
            row["opponent"],
            row["rollout_secs"],
            row["update_secs"],
            row["win_rate"],
            row["mean_reward"],
            row["policy_loss"],
            row["approx_kl"],
            row["bc_kl"],
        )
        # Save the latest snapshot every iter (local run_dir), then mirror to
        # S3 only when this iter improves the running best so the freshest
        # weights survive a mid-training host preempt (layer-1 safeguard).
        best_pt = run_dir / "best.pt"
        _save_best_pt(model, best_pt)
        # Also keep a per-iter local checkpoint so the sweet-spot model is
        # recoverable post-hoc. best.pt == latest, and self-play can overshoot
        # the generalizing optimum (it overfits to the current pool); a later
        # external eval (vs rl_v0 etc.) then picks the right iter's ckpt rather
        # than being stuck with the last (possibly degraded) model.
        _save_best_pt(model, run_dir / f"ckpt_i{it:03d}.pt")
        if row["win_rate"] >= best_win:
            best_win = row["win_rate"]
            _upload_best_to_s3(best_pt, row["iter"], best_win)

        # Flush metrics.json after EVERY iter (not just at the end) so the
        # reward/stat trace + opponent-pool distribution are readable mid-run
        # (tail / plot in flight). Pairs with the per-iter best.pt above so
        # both weights and stats are always retrievable before completion.
        _write_metrics(
            run_dir / "metrics.json",
            iterations_run=it + 1,
            total_episodes=(it + 1) * episodes_per_iter,
            best_win_rate=best_win,
            runtime_seconds=time.perf_counter() - started,
            history=history,
        )

    runtime = time.perf_counter() - started
    logger.info(
        "training done iters=%d runtime=%.1fs best_win=%.3f",
        iterations,
        runtime,
        best_win,
    )


if __name__ == "__main__":
    app()
