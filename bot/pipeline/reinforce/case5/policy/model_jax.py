"""JAX/Equinox port of `model.py` (ActorCritic) for end-to-end GPU rollout.

Mirrors the PyTorch architecture exactly so BC weights (case9 per_planet
+ value_head + ship_log_std) load 1:1 by parameter name:

  backbone        Set Transformer (ISAB × encoder_layers, hidden=192,
                  attn_heads=8, inducing_points=24)
  head_per_planet pointer-style (P+1) categorical + ship_pred regression
  value_head      2-layer MLP on the bottleneck context
  ship_log_std    state-independent scalar log-σ

Output shapes match `PolicyOutput`:
  per_planet_logits  (B, P, P+1)
  ship_mean          (B, P)
  ship_log_std       scalar
  value              (B,)

The model is an Equinox module (registered pytree), so the full forward
pass is `jit`/`vmap`-friendly. `load_bc_weights_jax(model, path)` reads a
PyTorch `.pt` file via `torch.load(..., weights_only=True)` and copies
weights leaf-by-leaf via numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .featurizer_jax import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
    BatchFeaturesJax,
)


@dataclass(frozen=True)
class ModelConfigJax:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 192
    attn_heads: int = 8
    inducing_points: int = 24
    encoder_layers: int = 4
    head_dropout: float = 0.0
    ship_log_std_init: float = 0.5
    no_op_bias: float = 8.0


class PolicyOutputJax(eqx.Module):
    per_planet_logits: jax.Array  # (B, P, P+1)
    ship_mean: jax.Array  # (B, P)
    ship_log_std: jax.Array  # scalar
    value: jax.Array  # (B,)


def _layer_norm_forward(
    x: jax.Array, weight: jax.Array, bias: jax.Array, eps: float = 1e-5
) -> jax.Array:
    """LayerNorm forward; PyTorch default eps=1e-5."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * weight + bias


def _multihead_attention(
    query: jax.Array,  # (B, Lq, H)
    key: jax.Array,  # (B, Lk, H)
    value: jax.Array,  # (B, Lk, H)
    in_proj_w: jax.Array,  # (3H, H)
    in_proj_b: jax.Array,  # (3H,)
    out_proj_w: jax.Array,  # (H, H)
    out_proj_b: jax.Array,  # (H,)
    num_heads: int,
    key_padding_mask: jax.Array | None,  # (B, Lk) bool — True = pad/ignore
) -> jax.Array:
    """Mirror torch.nn.MultiheadAttention(batch_first=True) forward.

    PyTorch packs Q/K/V projection weights as a single (3H, H) matrix in
    `in_proj_weight` for self-attention; for cross-attention with the
    same `embed_dim` this layout is preserved. We split it three ways.
    """
    h = query.shape[-1]
    head_dim = h // num_heads
    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))

    # Project to Q, K, V.
    qw, kw, vw = jnp.split(in_proj_w, 3, axis=0)  # each (H, H)
    qb, kb, vb = jnp.split(in_proj_b, 3, axis=0)  # each (H,)
    q = query @ qw.T + qb  # (B, Lq, H)
    k = key @ kw.T + kb  # (B, Lk, H)
    v = value @ vw.T + vb  # (B, Lk, H)

    # Split into heads.
    def _split(a: jax.Array) -> jax.Array:
        b, length, _ = a.shape
        return a.reshape(b, length, num_heads, head_dim).transpose(0, 2, 1, 3)

    qh = _split(q)  # (B, nh, Lq, dh)
    kh = _split(k)
    vh = _split(v)

    scores = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) * scale  # (B, nh, Lq, Lk)
    if key_padding_mask is not None:
        pad = key_padding_mask[:, None, None, :]  # (B, 1, 1, Lk)
        scores = jnp.where(pad, jnp.float32(-1e9), scores)
    attn = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, vh)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], h)
    out = ctx @ out_proj_w.T + out_proj_b
    return out


class MultiheadAttentionBlockJax(eqx.Module):
    mha_in_proj_weight: jax.Array  # (3H, H)
    mha_in_proj_bias: jax.Array  # (3H,)
    mha_out_proj_weight: jax.Array  # (H, H)
    mha_out_proj_bias: jax.Array  # (H,)
    ln1_w: jax.Array
    ln1_b: jax.Array
    ln2_w: jax.Array
    ln2_b: jax.Array
    ff0_w: jax.Array  # (2H, H)
    ff0_b: jax.Array  # (2H,)
    ff1_w: jax.Array  # (H, 2H)
    ff1_b: jax.Array  # (H,)
    num_heads: int = eqx.field(static=True)

    @staticmethod
    def from_init(
        key: jax.Array, dim: int, num_heads: int
    ) -> "MultiheadAttentionBlockJax":
        # Random init matching PyTorch's defaults (kaiming-ish but the
        # exact values are overwritten by BC weight load). We use a
        # cheap normal init since the test always loads BC anyway.
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return MultiheadAttentionBlockJax(
            mha_in_proj_weight=jax.random.normal(k1, (3 * dim, dim)) * 0.02,
            mha_in_proj_bias=jnp.zeros((3 * dim,), dtype=jnp.float32),
            mha_out_proj_weight=jax.random.normal(k2, (dim, dim)) * 0.02,
            mha_out_proj_bias=jnp.zeros((dim,), dtype=jnp.float32),
            ln1_w=jnp.ones((dim,), dtype=jnp.float32),
            ln1_b=jnp.zeros((dim,), dtype=jnp.float32),
            ln2_w=jnp.ones((dim,), dtype=jnp.float32),
            ln2_b=jnp.zeros((dim,), dtype=jnp.float32),
            ff0_w=jax.random.normal(k3, (dim * 2, dim)) * 0.02,
            ff0_b=jnp.zeros((dim * 2,), dtype=jnp.float32),
            ff1_w=jax.random.normal(k4, (dim, dim * 2)) * 0.02,
            ff1_b=jnp.zeros((dim,), dtype=jnp.float32),
            num_heads=num_heads,
        )

    def __call__(
        self,
        x: jax.Array,  # (B, Lx, H)
        y: jax.Array,  # (B, Ly, H)
        key_padding_mask: jax.Array | None,  # (B, Ly) bool — True = pad
    ) -> jax.Array:
        attn = _multihead_attention(
            x,
            y,
            y,
            self.mha_in_proj_weight,
            self.mha_in_proj_bias,
            self.mha_out_proj_weight,
            self.mha_out_proj_bias,
            self.num_heads,
            key_padding_mask,
        )
        h = _layer_norm_forward(x + attn, self.ln1_w, self.ln1_b)
        ff = jax.nn.relu(h @ self.ff0_w.T + self.ff0_b) @ self.ff1_w.T + self.ff1_b
        out = _layer_norm_forward(h + ff, self.ln2_w, self.ln2_b)
        return out


class InducedSetAttentionBlockJax(eqx.Module):
    inducing: jax.Array  # (1, num_inducing, dim)
    mab1: MultiheadAttentionBlockJax
    mab2: MultiheadAttentionBlockJax

    @staticmethod
    def from_init(
        key: jax.Array, dim: int, num_heads: int, num_inducing: int
    ) -> "InducedSetAttentionBlockJax":
        k1, k2, k3 = jax.random.split(key, 3)
        return InducedSetAttentionBlockJax(
            inducing=jax.random.normal(k1, (1, num_inducing, dim)) * 0.02,
            mab1=MultiheadAttentionBlockJax.from_init(k2, dim, num_heads),
            mab2=MultiheadAttentionBlockJax.from_init(k3, dim, num_heads),
        )

    def __call__(
        self,
        x: jax.Array,  # (B, P, H)
        mask: jax.Array,  # (B, P) bool — True = valid
    ) -> jax.Array:
        b = x.shape[0]
        i = jnp.broadcast_to(self.inducing, (b,) + self.inducing.shape[1:])
        kp_mask = ~mask  # True = pad (ignore)
        h = self.mab1(i, x, kp_mask)
        out = self.mab2(x, h, None)
        return out * mask[..., None].astype(jnp.float32)


class PMAJax(eqx.Module):
    seeds: jax.Array  # (1, num_seeds, dim)
    mab: MultiheadAttentionBlockJax

    @staticmethod
    def from_init(key: jax.Array, dim: int, num_heads: int, num_seeds: int) -> "PMAJax":
        k1, k2 = jax.random.split(key, 2)
        return PMAJax(
            seeds=jax.random.normal(k1, (1, num_seeds, dim)) * 0.02,
            mab=MultiheadAttentionBlockJax.from_init(k2, dim, num_heads),
        )

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        b = x.shape[0]
        s = jnp.broadcast_to(self.seeds, (b,) + self.seeds.shape[1:])
        kp_mask = ~mask
        return self.mab(s, x, kp_mask)


class SetTransformerBackboneJax(eqx.Module):
    in_proj_w: jax.Array  # (H, planet_in_dim)
    in_proj_b: jax.Array  # (H,)
    encoder: list[InducedSetAttentionBlockJax]
    pool: PMAJax
    psi_0_w: jax.Array  # (H, H + global_in_dim)
    psi_0_b: jax.Array
    psi_2_w: jax.Array  # (H, H)
    psi_2_b: jax.Array
    cfg: ModelConfigJax = eqx.field(static=True)

    @staticmethod
    def from_init(key: jax.Array, cfg: ModelConfigJax) -> "SetTransformerBackboneJax":
        h = cfg.hidden
        keys = jax.random.split(key, 6 + cfg.encoder_layers)
        encoder = [
            InducedSetAttentionBlockJax.from_init(
                keys[2 + i], h, cfg.attn_heads, cfg.inducing_points
            )
            for i in range(cfg.encoder_layers)
        ]
        return SetTransformerBackboneJax(
            in_proj_w=jax.random.normal(keys[0], (h, cfg.planet_in_dim)) * 0.02,
            in_proj_b=jnp.zeros((h,), dtype=jnp.float32),
            encoder=encoder,
            pool=PMAJax.from_init(keys[1], h, cfg.attn_heads, 1),
            psi_0_w=jax.random.normal(keys[-3], (h, h + cfg.global_in_dim)) * 0.02,
            psi_0_b=jnp.zeros((h,), dtype=jnp.float32),
            psi_2_w=jax.random.normal(keys[-2], (h, h)) * 0.02,
            psi_2_b=jnp.zeros((h,), dtype=jnp.float32),
            cfg=cfg,
        )

    def __call__(
        self,
        planet_feats: jax.Array,  # (B, P, F_planet)
        planet_mask: jax.Array,  # (B, P) bool
        global_feats: jax.Array,  # (B, F_global)
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (h_with_ctx, ctx, h)."""
        h0 = (planet_feats @ self.in_proj_w.T + self.in_proj_b) * planet_mask[
            ..., None
        ].astype(jnp.float32)
        h = h0
        for block in self.encoder:
            h = block(h, planet_mask)
        pooled = self.pool(h, planet_mask)[:, 0, :]  # (B, H)
        psi_in = jnp.concatenate([pooled, global_feats], axis=-1)
        psi_h = jax.nn.relu(psi_in @ self.psi_0_w.T + self.psi_0_b)
        ctx = psi_h @ self.psi_2_w.T + self.psi_2_b
        ctx_exp = jnp.broadcast_to(
            ctx[:, None, :], (ctx.shape[0], h.shape[1], ctx.shape[1])
        )
        h_with_ctx = jnp.concatenate([h, ctx_exp], axis=-1)
        return h_with_ctx, ctx, h


class PerPlanetHeadJax(eqx.Module):
    src_proj_0_w: jax.Array  # (H, 2H)
    src_proj_0_b: jax.Array
    src_proj_2_w: jax.Array  # (H, H)
    src_proj_2_b: jax.Array
    tgt_proj_0_w: jax.Array
    tgt_proj_0_b: jax.Array
    tgt_proj_2_w: jax.Array
    tgt_proj_2_b: jax.Array
    pair_proj_w: jax.Array  # (H, H), no bias
    noop_query: jax.Array  # (1, 1, H)
    ship_head_0_w: jax.Array  # (H, 2H)
    ship_head_0_b: jax.Array
    ship_head_2_w: jax.Array  # (1, H)
    ship_head_2_b: jax.Array  # (1,)

    @staticmethod
    def from_init(key: jax.Array, hidden: int) -> "PerPlanetHeadJax":
        h = hidden
        keys = jax.random.split(key, 11)

        def w(k: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            return jax.random.normal(k, shape) * 0.02

        return PerPlanetHeadJax(
            src_proj_0_w=w(keys[0], (h, 2 * h)),
            src_proj_0_b=jnp.zeros((h,), dtype=jnp.float32),
            src_proj_2_w=w(keys[1], (h, h)),
            src_proj_2_b=jnp.zeros((h,), dtype=jnp.float32),
            tgt_proj_0_w=w(keys[2], (h, 2 * h)),
            tgt_proj_0_b=jnp.zeros((h,), dtype=jnp.float32),
            tgt_proj_2_w=w(keys[3], (h, h)),
            tgt_proj_2_b=jnp.zeros((h,), dtype=jnp.float32),
            pair_proj_w=w(keys[4], (h, h)),
            noop_query=w(keys[5], (1, 1, h)),
            ship_head_0_w=w(keys[6], (h, 2 * h)),
            ship_head_0_b=jnp.zeros((h,), dtype=jnp.float32),
            ship_head_2_w=w(keys[7], (1, h)),
            ship_head_2_b=jnp.zeros((1,), dtype=jnp.float32),
        )

    def __call__(
        self,
        h_with_ctx: jax.Array,  # (B, P, 2H)
        planet_mask: jax.Array,  # (B, P) bool
    ) -> tuple[jax.Array, jax.Array]:
        src_h = jax.nn.relu(h_with_ctx @ self.src_proj_0_w.T + self.src_proj_0_b)
        src_q = src_h @ self.src_proj_2_w.T + self.src_proj_2_b
        tgt_h = jax.nn.relu(h_with_ctx @ self.tgt_proj_0_w.T + self.tgt_proj_0_b)
        tgt_k = tgt_h @ self.tgt_proj_2_w.T + self.tgt_proj_2_b
        tgt_kp = tgt_k @ self.pair_proj_w.T  # no bias
        pair_logits = jnp.einsum("bph,bqh->bpq", src_q, tgt_kp)
        # No-op logit per src.
        noop_q = jnp.broadcast_to(self.noop_query, (src_q.shape[0], 1, src_q.shape[-1]))
        noop_logits = jnp.einsum("bph,bqh->bpq", src_q, noop_q)  # (B, P, 1)
        # Mask invalid targets.
        mask = jnp.broadcast_to(planet_mask[:, None, :], pair_logits.shape)
        pair_logits = jnp.where(mask, pair_logits, jnp.float32(-1e9))
        per_planet_logits = jnp.concatenate([pair_logits, noop_logits], axis=-1)
        # Ships regression.
        ship_h = jax.nn.relu(h_with_ctx @ self.ship_head_0_w.T + self.ship_head_0_b)
        ship_pred = (ship_h @ self.ship_head_2_w.T + self.ship_head_2_b).squeeze(-1)
        return per_planet_logits, ship_pred


class ValueHeadJax(eqx.Module):
    net_0_w: jax.Array  # (H, H)
    net_0_b: jax.Array
    net_2_w: jax.Array  # (1, H)
    net_2_b: jax.Array

    @staticmethod
    def from_init(key: jax.Array, hidden: int) -> "ValueHeadJax":
        k1, k2 = jax.random.split(key, 2)
        return ValueHeadJax(
            net_0_w=jax.random.normal(k1, (hidden, hidden)) * 0.02,
            net_0_b=jnp.zeros((hidden,), dtype=jnp.float32),
            net_2_w=jax.random.normal(k2, (1, hidden)) * 0.02,
            net_2_b=jnp.zeros((1,), dtype=jnp.float32),
        )

    def __call__(self, ctx: jax.Array) -> jax.Array:
        h = jax.nn.relu(ctx @ self.net_0_w.T + self.net_0_b)
        return (h @ self.net_2_w.T + self.net_2_b).squeeze(-1)


class ActorCriticJax(eqx.Module):
    backbone: SetTransformerBackboneJax
    head_per_planet: PerPlanetHeadJax
    value_head: ValueHeadJax
    ship_log_std: jax.Array  # (1,)
    cfg: ModelConfigJax = eqx.field(static=True)

    @staticmethod
    def from_init(
        key: jax.Array, cfg: ModelConfigJax | None = None
    ) -> "ActorCriticJax":
        cfg = cfg or ModelConfigJax()
        k1, k2, k3 = jax.random.split(key, 3)
        return ActorCriticJax(
            backbone=SetTransformerBackboneJax.from_init(k1, cfg),
            head_per_planet=PerPlanetHeadJax.from_init(k2, cfg.hidden),
            value_head=ValueHeadJax.from_init(k3, cfg.hidden),
            ship_log_std=jnp.full(
                (1,), jnp.float32(cfg.ship_log_std_init), dtype=jnp.float32
            ),
            cfg=cfg,
        )

    def __call__(self, batch: BatchFeaturesJax) -> PolicyOutputJax:
        h_with_ctx, ctx, _ = self.backbone(
            batch.planet_feats, batch.planet_mask, batch.global_feats
        )
        per_planet_logits, ship_mean = self.head_per_planet(
            h_with_ctx, batch.planet_mask
        )
        if self.cfg.no_op_bias != 0.0:
            offset = jnp.zeros_like(per_planet_logits)
            offset = offset.at[..., -1].set(-self.cfg.no_op_bias)
            per_planet_logits = per_planet_logits + offset
        value = self.value_head(ctx)
        return PolicyOutputJax(
            per_planet_logits=per_planet_logits,
            ship_mean=ship_mean,
            ship_log_std=self.ship_log_std,
            value=value,
        )


def _torch_to_jax(t: "Any") -> jax.Array:  # type: ignore[name-defined]  # noqa: F821
    """torch.Tensor → jax.Array via numpy."""
    return jnp.asarray(np.asarray(t.detach().cpu().numpy(), dtype=np.float32))


def load_bc_weights_jax(
    model: ActorCriticJax, weights_path: str
) -> tuple[ActorCriticJax, int, int]:
    """Replace `model`'s weight leaves from a PyTorch BC `.pt` state_dict.

    Returns (new_model, loaded_count, missing_count). `missing` reflects
    JAX leaves whose PyTorch counterpart wasn't found (e.g. value_head,
    ship_log_std on a clean BC warm-start). Mirrors the PyTorch
    `load_bc_weights` contract.

    Mapping convention:
      backbone.in_proj.{weight,bias}              → in_proj_w/b
      backbone.encoder.<i>.mab{1,2}.mha.in_proj_*  → encoder[i].mab{1,2}.mha_in_proj_*
                            .mha.out_proj.*       → ...mha_out_proj_*
                            .ln{1,2}.{weight,bias} → ln{1,2}_w/b
                            .ff.{0,2}.{weight,bias} → ff{0,1}_w/b  (note 2→1 collapse)
      backbone.encoder.<i>.inducing              → encoder[i].inducing
      backbone.pool.seeds                         → pool.seeds
      backbone.pool.mab.*                         → pool.mab.*
      backbone.psi.{0,2}.{weight,bias}            → psi_{0,2}_w/b
      head_per_planet.src_proj.{0,3}.{w,b}        → src_proj_{0,2}_w/b
      head_per_planet.tgt_proj.{0,3}.{w,b}        → tgt_proj_{0,2}_w/b
      head_per_planet.pair_proj.weight            → pair_proj_w
      head_per_planet.noop_query                  → noop_query
      head_per_planet.ship_head.{0,3}.{w,b}       → ship_head_{0,2}_w/b
      value_head.net.{0,2}.{w,b}                  → value_head.net_{0,2}_w/b
      ship_log_std.log_std                        → ship_log_std

    Note on PyTorch nn.Sequential indices: the `nn.ReLU` and `nn.Dropout`
    layers occupy positions in the index sequence; the linear layers are
    at indices 0 and 2 (or 0 and 3 if a Dropout sits between). For the
    case9 BC weights the relevant linear positions are 0 and 2 (no
    dropout) for backbone.psi / value_head.net, and 0 and 3 for the head
    src/tgt/ship MLPs (head_dropout layer at index 2).
    """
    import torch

    state: dict[str, "torch.Tensor"] = torch.load(
        weights_path, map_location="cpu", weights_only=True
    )

    loaded = 0
    missing = 0

    # ---- mapping helpers ----
    def get(key: str) -> jax.Array | None:
        nonlocal loaded
        if key in state:
            loaded += 1
            return _torch_to_jax(state[key])
        return None

    def gor(key: str, fallback: jax.Array) -> jax.Array:
        """get(key) if present else fallback. Use this instead of `or`
        because jax arrays don't have a bool conversion."""
        got = get(key)
        return got if got is not None else fallback

    # Backbone in_proj.
    in_proj_w = get("backbone.in_proj.weight")
    in_proj_b = get("backbone.in_proj.bias")

    encoder = list(model.backbone.encoder)
    for i, block in enumerate(encoder):
        prefix = f"backbone.encoder.{i}"
        inducing = get(f"{prefix}.inducing")

        def load_mab(
            mab_prefix: str, mab: MultiheadAttentionBlockJax
        ) -> MultiheadAttentionBlockJax:
            return MultiheadAttentionBlockJax(
                mha_in_proj_weight=gor(
                    f"{mab_prefix}.mha.in_proj_weight", mab.mha_in_proj_weight
                ),
                mha_in_proj_bias=gor(
                    f"{mab_prefix}.mha.in_proj_bias", mab.mha_in_proj_bias
                ),
                mha_out_proj_weight=gor(
                    f"{mab_prefix}.mha.out_proj.weight", mab.mha_out_proj_weight
                ),
                mha_out_proj_bias=gor(
                    f"{mab_prefix}.mha.out_proj.bias", mab.mha_out_proj_bias
                ),
                ln1_w=gor(f"{mab_prefix}.ln1.weight", mab.ln1_w),
                ln1_b=gor(f"{mab_prefix}.ln1.bias", mab.ln1_b),
                ln2_w=gor(f"{mab_prefix}.ln2.weight", mab.ln2_w),
                ln2_b=gor(f"{mab_prefix}.ln2.bias", mab.ln2_b),
                ff0_w=gor(f"{mab_prefix}.ff.0.weight", mab.ff0_w),
                ff0_b=gor(f"{mab_prefix}.ff.0.bias", mab.ff0_b),
                ff1_w=gor(f"{mab_prefix}.ff.2.weight", mab.ff1_w),
                ff1_b=gor(f"{mab_prefix}.ff.2.bias", mab.ff1_b),
                num_heads=mab.num_heads,
            )

        encoder[i] = InducedSetAttentionBlockJax(
            inducing=inducing if inducing is not None else block.inducing,
            mab1=load_mab(f"{prefix}.mab1", block.mab1),
            mab2=load_mab(f"{prefix}.mab2", block.mab2),
        )

    # PMA pool.
    pool_seeds = get("backbone.pool.seeds")
    pool_mab = model.backbone.pool.mab
    new_pool_mab = MultiheadAttentionBlockJax(
        mha_in_proj_weight=gor(
            "backbone.pool.mab.mha.in_proj_weight", pool_mab.mha_in_proj_weight
        ),
        mha_in_proj_bias=gor(
            "backbone.pool.mab.mha.in_proj_bias", pool_mab.mha_in_proj_bias
        ),
        mha_out_proj_weight=gor(
            "backbone.pool.mab.mha.out_proj.weight", pool_mab.mha_out_proj_weight
        ),
        mha_out_proj_bias=gor(
            "backbone.pool.mab.mha.out_proj.bias", pool_mab.mha_out_proj_bias
        ),
        ln1_w=gor("backbone.pool.mab.ln1.weight", pool_mab.ln1_w),
        ln1_b=gor("backbone.pool.mab.ln1.bias", pool_mab.ln1_b),
        ln2_w=gor("backbone.pool.mab.ln2.weight", pool_mab.ln2_w),
        ln2_b=gor("backbone.pool.mab.ln2.bias", pool_mab.ln2_b),
        ff0_w=gor("backbone.pool.mab.ff.0.weight", pool_mab.ff0_w),
        ff0_b=gor("backbone.pool.mab.ff.0.bias", pool_mab.ff0_b),
        ff1_w=gor("backbone.pool.mab.ff.2.weight", pool_mab.ff1_w),
        ff1_b=gor("backbone.pool.mab.ff.2.bias", pool_mab.ff1_b),
        num_heads=pool_mab.num_heads,
    )
    new_pool = PMAJax(
        seeds=pool_seeds if pool_seeds is not None else model.backbone.pool.seeds,
        mab=new_pool_mab,
    )

    # Backbone psi MLP.
    psi_0_w = get("backbone.psi.0.weight")
    psi_0_b = get("backbone.psi.0.bias")
    psi_2_w = get("backbone.psi.2.weight")
    psi_2_b = get("backbone.psi.2.bias")

    new_backbone = SetTransformerBackboneJax(
        in_proj_w=in_proj_w if in_proj_w is not None else model.backbone.in_proj_w,
        in_proj_b=in_proj_b if in_proj_b is not None else model.backbone.in_proj_b,
        encoder=encoder,
        pool=new_pool,
        psi_0_w=psi_0_w if psi_0_w is not None else model.backbone.psi_0_w,
        psi_0_b=psi_0_b if psi_0_b is not None else model.backbone.psi_0_b,
        psi_2_w=psi_2_w if psi_2_w is not None else model.backbone.psi_2_w,
        psi_2_b=psi_2_b if psi_2_b is not None else model.backbone.psi_2_b,
        cfg=model.backbone.cfg,
    )

    # Per-planet head. The PyTorch nn.Sequential indices are 0/3 because
    # of an interleaved nn.Dropout at index 2.
    new_head = PerPlanetHeadJax(
        src_proj_0_w=gor(
            "head_per_planet.src_proj.0.weight", model.head_per_planet.src_proj_0_w
        ),
        src_proj_0_b=gor(
            "head_per_planet.src_proj.0.bias", model.head_per_planet.src_proj_0_b
        ),
        src_proj_2_w=gor(
            "head_per_planet.src_proj.3.weight", model.head_per_planet.src_proj_2_w
        ),
        src_proj_2_b=gor(
            "head_per_planet.src_proj.3.bias", model.head_per_planet.src_proj_2_b
        ),
        tgt_proj_0_w=gor(
            "head_per_planet.tgt_proj.0.weight", model.head_per_planet.tgt_proj_0_w
        ),
        tgt_proj_0_b=gor(
            "head_per_planet.tgt_proj.0.bias", model.head_per_planet.tgt_proj_0_b
        ),
        tgt_proj_2_w=gor(
            "head_per_planet.tgt_proj.3.weight", model.head_per_planet.tgt_proj_2_w
        ),
        tgt_proj_2_b=gor(
            "head_per_planet.tgt_proj.3.bias", model.head_per_planet.tgt_proj_2_b
        ),
        pair_proj_w=gor(
            "head_per_planet.pair_proj.weight", model.head_per_planet.pair_proj_w
        ),
        noop_query=gor("head_per_planet.noop_query", model.head_per_planet.noop_query),
        ship_head_0_w=gor(
            "head_per_planet.ship_head.0.weight", model.head_per_planet.ship_head_0_w
        ),
        ship_head_0_b=gor(
            "head_per_planet.ship_head.0.bias", model.head_per_planet.ship_head_0_b
        ),
        ship_head_2_w=gor(
            "head_per_planet.ship_head.3.weight", model.head_per_planet.ship_head_2_w
        ),
        ship_head_2_b=gor(
            "head_per_planet.ship_head.3.bias", model.head_per_planet.ship_head_2_b
        ),
    )

    # Value head (BC weights almost never include this; it's added by
    # PPO. Allow it to be missing.).
    new_value = ValueHeadJax(
        net_0_w=gor("value_head.net.0.weight", model.value_head.net_0_w),
        net_0_b=gor("value_head.net.0.bias", model.value_head.net_0_b),
        net_2_w=gor("value_head.net.2.weight", model.value_head.net_2_w),
        net_2_b=gor("value_head.net.2.bias", model.value_head.net_2_b),
    )

    new_ship_log_std = get("ship_log_std.log_std")
    if new_ship_log_std is None:
        new_ship_log_std = model.ship_log_std

    new_model = ActorCriticJax(
        backbone=new_backbone,
        head_per_planet=new_head,
        value_head=new_value,
        ship_log_std=new_ship_log_std,
        cfg=model.cfg,
    )
    # Count missing: total leaves that get() didn't satisfy. We can
    # approximate by inspecting the BC state_dict key count vs how many
    # we attempted (some keys we explicitly don't map, e.g. extras).
    expected = len(state)
    missing = max(0, expected - loaded)
    return new_model, loaded, missing


__all__ = [
    "ModelConfigJax",
    "PolicyOutputJax",
    "ActorCriticJax",
    "load_bc_weights_jax",
]
