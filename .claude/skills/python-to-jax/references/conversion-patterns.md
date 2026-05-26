# Python → JAX Conversion Patterns

Concrete before/after translations, grouped by the construct you're replacing.
Examples are drawn from this repo's real ports (`jax_env/`, the `*_jax.py`
files under `bot/pipeline/reinforce/case1`). Read the matching source file when
you need more context — these are distillations, not the whole story.

## Table of contents

1. [Dynamic length → fixed shape + mask](#1-dynamic-length--fixed-shape--mask)
2. [Python loops → lax.scan / vmap](#2-python-loops--laxscan--vmap)
3. [Conditionals → lax.cond / jnp.where](#3-conditionals--laxcond--jnpwhere)
4. [Mutation → immutable updates](#4-mutation--immutable-updates)
5. [PyTorch nn.Module → equinox.Module](#5-pytorch-nnmodule--equinoxmodule)
6. [PyTorch optimizer/loss → Optax](#6-pytorch-optimizerloss--optax)
7. [Weight loading by name](#7-weight-loading-by-name)
8. [Avoiding recompilation (one trace)](#8-avoiding-recompilation-one-trace)
9. [Host-side vs device-side split](#9-host-side-vs-device-side-split)
10. [Randomness: PRNGKey threading](#10-randomness-prngkey-threading)
11. [State containers: pytrees](#11-state-containers-pytrees)
12. [The hard hot-path constructs (refinement loops, variable-length search, ragged dispatch)](#12-the-hard-hot-path-constructs)

---

## 1. Dynamic length → fixed shape + mask

JAX compiles for a fixed shape. Anything whose size depends on data (a list that
grows, `arr[mask]` boolean indexing, a variable loop count) cannot be traced.
The fix: pad to a `MAX_*` constant, carry a boolean validity mask, compute over
the full array, and neutralize padding with `jnp.where`.

```python
# Python — variable-length, owner-filtered sum
def my_ships(planets, seat):
    return sum(p.ships for p in planets if p.owner == seat)
```
```python
# JAX — fixed MAX_PLANETS, masked reduce
def my_ships(state: EnvState, seat: int) -> jax.Array:
    is_mine = state.planet_valid & (state.planet_owner == seat)
    return jnp.sum(jnp.where(is_mine, state.planet_ships, 0.0))
```

Key idea: `planet_valid` distinguishes real slots from padding; you always
operate on the full `(MAX_PLANETS,)` array. See `rollout_jax.py::_ship_totals`.

## 2. Python loops → lax.scan / vmap

A sequential loop with a carried state → `lax.scan`. Independent parallel
repeats → `vmap`.

```python
# Python — sequential rollout
state = reset(seed)
transitions = []
for t in range(horizon):
    action = policy(state)
    state, reward = env.step(state, action)
    transitions.append((action, reward))
```
```python
# JAX — lax.scan: carry threads state, per-step outputs stack into (T, ...)
def step_fn(carry, _t):
    state, key = carry
    key, k = jax.random.split(key)
    action = policy(state, k)
    new_state, reward = env_step(state, action)
    return (new_state, key), (action, reward)

(final_state, _), (actions_t, rewards_t) = jax.lax.scan(
    step_fn, (init_state, key), jnp.arange(horizon)
)
```

```python
# N parallel episodes — vmap over the per-episode args, broadcast the model.
vmapped = jax.vmap(_rollout_one_env, in_axes=(None, 0, 0, None))
#                                              ^model ^key ^state ^scalar
batch = vmapped(model, keys, batched_init_states, horizon)
```

`scan` requires the carry pytree to keep identical shape/dtype every iteration —
this is *why* env state must be fixed-shape (pattern 1). To "freeze" a finished
episode while peers keep running, don't break — gate with a `done` flag:
`state_next = jax.tree.map(lambda new, old: jnp.where(done, old, new), new, old)`.
See `rollout_jax.py::_rollout_one_env`.

Nested loops (PPO epochs × minibatches) → nested `scan`. See
`ppo_jax.py::ppo_update_jax`, which scans epochs over scanned minibatches and
even implements `target_kl` early-stop as a bool flag in the carry (no
data-dependent break).

## 3. Conditionals → lax.cond / jnp.where

`if` on a *static* Python value is fine and stays in Python. `if` on a *traced*
value must become `lax.cond` (when the branches are expensive / have side
structure) or `jnp.where` (cheap, elementwise — usually preferred).

```python
if cnt > 0: adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # Python (cnt static? keep it)
```
```python
# traced predicate, two whole-array branches → lax.cond
diff = jax.lax.cond(mode == PLANETS, lambda: plt_my - plt_en, lambda: ship_my - ship_en)
# elementwise select → jnp.where
reward = jnp.where(done_already, 0.0, shaping + terminal)
```

See `rollout_jax.py::_shaping_diff` (`lax.cond`) and the `jnp.where` terminal-
reward gating in the same file.

## 4. Mutation → immutable updates

JAX arrays are immutable. In-place ops become functional updates.

| Python | JAX |
|---|---|
| `arr[i] = v` | `arr = arr.at[i].set(v)` |
| `arr[i] += v` | `arr = arr.at[i].add(v)` |
| `state.x = v` (dataclass) | `state = state._replace(x=v)` (NamedTuple) or `eqx.tree_at` |
| `list.append(x)` in loop | collect via `scan` stacked outputs |
| `dict[k] = v` accumulator | return an aux dict from the scan body |

Never write `NEVER mutate` violations — this repo's Python rules already ban
mutation, and JAX makes it a hard error.

## 5. PyTorch nn.Module → equinox.Module

```python
# PyTorch
class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value_head = nn.Linear(cfg.hidden, 1)
    def forward(self, x):
        return self.value_head(x)
```
```python
# Equinox — fields are the pytree leaves; __call__ is the forward
class ActorCriticJax(eqx.Module):
    value_w: jax.Array
    value_b: jax.Array
    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.value_w.T + self.value_b
    @classmethod
    def from_init(cls, key: jax.Array) -> "ActorCriticJax":
        ...  # initialize leaves with jax.random
```

Because the module is a registered pytree, `model(batch)` is jit/vmap-able and
`eqx.filter_value_and_grad(loss_fn)(model, ...)` differentiates w.r.t. the array
leaves only (filtering out static config). Output containers are also
`eqx.Module`s or `NamedTuple`s (see `PolicyOutputJax`). Mirror the PyTorch
architecture *exactly* — same layer order, dims, presence/absence of bias — or
weight loading (pattern 7) silently mismatches. See `model_jax.py`.

## 6. PyTorch optimizer/loss → Optax

```python
# PyTorch
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
opt.step(); opt.zero_grad()
```
```python
# Optax — clip then adamw, threaded functionally
def make_optimizer(cfg) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),   # == clip_grad_norm_
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )

grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
(loss, aux), grads = grad_fn(model, ...)
updates, opt_state = optimizer.update(grads, opt_state, model)
model = eqx.apply_updates(model, updates)
```

`opt_state` is threaded explicitly (no hidden mutable optimizer object). For an
lr schedule, pass an `optax.Schedule` as `learning_rate`. Loss math translates
directly: `nn.functional.log_softmax`→`jax.nn.log_softmax`,
`F.mse_loss(a,b)`→`jnp.mean((a-b)**2)`, `.mean()`→`jnp.mean(...)`,
`torch.clamp`→`jnp.clip`, `torch.exp`→`jnp.exp`. See `ppo_jax.py`.

## 7. Weight loading by name

A ported model is useless if it can't ingest the trained `.pt`. Read PyTorch
weights and copy leaf-by-leaf by parameter name, then **assert nothing was left
unmapped**.

```python
def load_bc_weights_jax(model, path):
    sd = torch.load(path, weights_only=True)          # never unpickle arbitrary code
    loaded, missing = 0, 0
    # map each torch key -> the corresponding eqx leaf, copy via numpy, count.
    ...
    return model, loaded, missing
# in the parity test:
model, loaded, missing = load_bc_weights_jax(model, weights_path)
assert missing == 0, f"{missing} torch keys unmapped"
```

`weights_only=True` is mandatory (no arbitrary code execution from a pickle).
See `model_jax.py::load_bc_weights_jax` and its parity test.

## 8. Avoiding recompilation (one trace)

jit recompiles for every distinct *static* argument and every new *shape*. A
Python `int seed` threaded into a jitted body recompiles per seed — this repo
blew its compile cache past 3 GB and hit SIGABRT exactly this way.

```python
# BAD — recompiles per seed value, one trace per episode
def rollout(model, seed: int, horizon): state = reset(seed); ...

# GOOD — build states on host, stack into a batched pytree, pass the array
init_states = [reset(seed=seed + i) for i in range(n)]          # host loop
batched = jax.tree.map(lambda *xs: jnp.stack(xs), *init_states)  # (n, ...)
vmapped = jax.vmap(rollout_one, in_axes=(None, 0, 0, None))      # one trace
```

Rules of thumb: keep varying scalars as traced arrays (or `jnp.int32(x)`), keep
truly-constant config as `static`/closure, and keep all batch elements the same
shape. See `rollout_jax.py::collect_rollout_jax`.

## 9. Host-side vs device-side split

Not everything should be jitted. Rejection sampling, RNG that must bit-match a
reference simulator, and variable-count generation belong on the **host** (plain
NumPy/Python), run once at `reset`, then `jnp.array`-ified into the state. The
per-step `step()` is the pure-jax, jitted hot path. This repo precomputes all
comet spawn paths at reset (host) so `step` never does rejection sampling. State
the split explicitly in the file docstring. See `jax_env/reset.py` (host) vs
`jax_env/step.py` (jit).

## 10. Randomness: PRNGKey threading

PyTorch/NumPy global RNG → explicit JAX keys. Split before every consuming call;
never reuse a key.

```python
key, k_sample = jax.random.split(key)   # thread `key` forward in the carry
action = sample_action_jax(output, mask, k_sample)
```
Per-episode keys come from `jax.random.split(key, n)`. A reused key gives
identical "random" draws — a subtle parity bug.

## 11. State containers: pytrees

Group related arrays into a `NamedTuple` (transitions, configs, rollout batches)
or a registered pytree class (`EnvState`). Pytrees flow through `scan`/`vmap`/
`tree.map` as one object. Use `_replace(field=...)` for functional updates and
`jax.tree.map(fn, a, b)` for elementwise ops across two same-structured trees
(e.g. the done-gating freeze in pattern 2). See `state.py::EnvState`,
`rollout_jax.py::JaxRolloutBatch`, `ppo_jax.py::FlatRollout`.

## 12. The hard hot-path constructs

These are the patterns that make agent/solver hot paths "hard to JAX" — and
exactly the ones worth porting, because a hot path is hard *because* it's doing
the per-turn work. All three are demonstrated together in
`bot/pipeline/rulebase/case2/baseline_jax/aim_jax.py` (a port of the per-(src,
target) intercept solver `aim_with_prediction` / `search_safe_intercept`),
parity-tested in `test_aim_jax_parity.py`. Read that file alongside this.

The framing that makes them tractable: don't port the function to run once —
port it so it can be **`vmap`'d over the entire loop domain** (every (src,target)
pair) in one call, replacing the Python `for src: for target:` double loop. That
batched call is where the speedup comes from.

### 12a. Fixed-iteration refinement with early-exit

Python: `for _ in range(5):` that mutates an estimate and `return`s early once it
converges. You can't `break` in a traced loop. Use `lax.scan` over the fixed
iteration count, carry a `done` bool, and **freeze updates once done** instead of
breaking — the carry stops changing, so later iterations are no-ops but the trace
stays static. Read the answer from the final carry.

```python
def refine_step(carry, _):
    tx, ty, ang, turns, done, valid, fellback = carry
    px, py = predict_lead(...); n_ang, n_turns, n_ok = estimate(...)
    converged = (jnp.abs(px - tx) < EPS) & (jnp.abs(py - ty) < EPS) & ...
    active = ~done & n_ok                       # only live, valid steps update
    tx = jnp.where(active, px, tx)              # freeze when ~active
    done = done | (active & converged) | (~n_ok)
    fellback = fellback | (~done_prev & ~n_ok)  # invalid est -> use the sweep path
    return (tx, ty, n_ang_or_old, ..., done, valid, fellback), None
carry, _ = jax.lax.scan(refine_step, init, jnp.arange(REFINE_ITERS))
```

Annotate the carry tuple with a named type alias (`RefineCarry = tuple[jax.Array,
...]` spelled out) so mypy accepts the `scan` signature.

### 12b. Variable-length search + early-exit → fixed grid + masked argmin

Python: `for candidate in range(1, max_turns+1):` building a `best` by comparing a
score tuple, where `max_turns` varies per call. Replace the variable bound with a
**fixed candidate grid** (`MAX_CANDIDATE_TURNS`), `vmap` the per-candidate score
over it, mask out-of-range / invalid candidates to `+inf`, then `argmin`.

```python
CAND = jnp.arange(1, MAX_CANDIDATE_TURNS + 1, dtype=jnp.float32) * STEP
def score_one(cand):
    ...                                   # compute candidate result + validity
    ok = valid & consistent & (cand <= max_turns)
    score = delta*1e6 + turns*1e2 + cand  # fold the Python score tuple to one
    return jnp.where(ok, score, jnp.inf), angle, turns, ix, iy
scores, *cols = jax.vmap(score_one)(CAND)
best = jnp.argmin(scores)
any_valid = jnp.isfinite(scores[best])    # all +inf -> Python `None`
```

The lexicographic Python score tuple `(delta, turns, candidate)` folds to one
float by scaling each level so it dominates the next (`delta*1e6 + turns*1e2 +
cand`) — pick scales larger than the next field's max range. `any_valid` is the
`None` sentinel.

### 12c. Ragged, data-dependent dispatch → resolve on host, compute in JAX

A branch like `if target.id in comet_ids: predict_comet_position(...)` reads a
variable list of comet-group dicts with `list.index` and ragged per-comet path
arrays. The instinct is "ragged → leave it in Python." Resist that instinct when
the branch is on the hot path — **separate the ragged *lookup* from the numeric
*computation*.** The lookup (which group, which path, the path_index) is cheap,
non-vectorizable host bookkeeping; do it once on the host and **resolve it into a
fixed-shape array**: pad each target's path to `(MAX_COMET_PATH_LEN, 2)` and pass
`(path, path_index, path_len)` as arrays. Then the numeric part runs in JAX for
*every* target, comet or not — the comet branch becomes a bounds-masked gather:

```python
# host: resolve the ragged dict -> fixed arrays (per target)
def resolve_comet_path(target_id, comets, comet_ids) -> (path_padded, path_index, path_len): ...
#   non-comet target -> path_len = 0  (signals the orbital branch in JAX)

# JAX: predict_comet_position as a clamped gather + ok flag (None -> ok=False)
future_idx = path_index + turn
ok = (future_idx >= 0) & (future_idx < path_len)
pt = comet_path[jnp.clip(future_idx, 0, MAX_LEN - 1)]
# dispatch on path_len>0; orbital branch otherwise — both in the same vmap.
```

This is strictly better than host-deferring the whole branch: comet targets now
ride the same `vmap` as everyone else, and the parity test covers them too. See
`aim_jax.py::resolve_comet_path` / `_predict_comet_fractional` and the comet case
in `test_aim_jax_parity.py`.

When is host-only (no port at all) actually right? When the branch is *not* on
the hot path — rolling opponent-model state updated once per turn, replay
logging, RNG that must bit-match a reference sim. The litmus test: would
vectorizing it speed up the entry point? If yes, host-resolve + JAX-compute it
(above). If no, host is fine — but say so, and confirm it isn't secretly the
bottleneck you're dodging (§ scope-by-hot-path in SKILL.md).
