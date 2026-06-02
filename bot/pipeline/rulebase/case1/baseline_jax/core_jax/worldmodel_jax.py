"""JAX-native port of baseline/core/world_model.py defense layer.

Covers resolve_arrival_event (combat), simulate_planet_timeline (110-turn fold),
and keep_needed (min surviving garrison). Faithful to the Python originals;
fixed-shape (arrivals padded to MAX_ARR, owners indexed 0..NUM_OWNERS-1 with a
neutral slot), jit/vmap-friendly.

resolve_arrival_event semantics (world_model.py:98-132):
  aggregate incoming ships per owner; top-2 by ships. tie → neutralized
  (survivor owner -1, ships 0). survivor takes (top - second) ships. if
  survivor == current owner: garrison += survivor_ships. else garrison -=
  survivor_ships; if garrison < 0 → owner flips to survivor with -garrison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Arr = jax.Array

# Owner encoding for combat: index 0..NUM_PLAYERS-1 are players; neutral = -1.
NUM_PLAYERS = 4


def resolve_arrival_event(
    owner: Arr, garrison: Arr, by_owner_ships: Arr
) -> tuple[Arr, Arr]:
    """Resolve one combat tick.

    by_owner_ships: float32[NUM_PLAYERS] — total incoming ships for each player
    index (0..NUM_PLAYERS-1). Neutral arrivals never occur (owner != -1 for
    fleets), so no neutral attacker slot is needed.

    Returns (new_owner, new_garrison). new_owner in {-1, 0..NUM_PLAYERS-1}.
    """
    total = jnp.sum(by_owner_ships)

    # top-1 and top-2 by ships (ties broken by lower index via stable argsort).
    order = jnp.argsort(-by_owner_ships, stable=True)  # descending
    top_owner = order[0]
    top_ships = by_owner_ships[top_owner]
    second_ships = by_owner_ships[order[1]]

    # number of distinct attackers (owners with >0 ships)
    n_attackers = jnp.sum(by_owner_ships > 0)

    tie = (n_attackers > 1) & (top_ships == second_ships)
    multi = n_attackers > 1
    survivor_owner = jnp.where(tie, -1, top_owner)
    survivor_ships = jnp.where(
        tie, 0.0, jnp.where(multi, top_ships - second_ships, top_ships)
    )

    no_event = (total <= 0) | (survivor_ships <= 0)

    # survivor == current owner → reinforce
    same = survivor_owner == owner
    g_reinforce = garrison + survivor_ships
    # else: garrison -= survivor_ships; flip if < 0
    g_after = garrison - survivor_ships
    flipped = g_after < 0
    out_owner = jnp.where(same, owner, jnp.where(flipped, survivor_owner, owner))
    out_garrison = jnp.where(same, g_reinforce, jnp.where(flipped, -g_after, g_after))

    final_owner = jnp.where(no_event, owner, out_owner).astype(jnp.int32)
    final_garrison = jnp.where(no_event, jnp.maximum(0.0, garrison), out_garrison)
    return final_owner, jnp.maximum(0.0, final_garrison)


def _per_turn_owner_ships(
    arr_eta: Arr, arr_owner: Arr, arr_ships: Arr, turn: Arr
) -> Arr:
    """Aggregate arrivals landing exactly on `turn` into a [NUM_PLAYERS] vector."""
    hit = (arr_eta == turn) & (arr_ships > 0)
    # one-hot over owners, weighted by ships, masked to this turn
    onehot = jax.nn.one_hot(arr_owner, NUM_PLAYERS) * (arr_ships * hit)[:, None]
    return jnp.sum(onehot, axis=0)


def _simulate(
    owner0: Arr,
    garrison0: Arr,
    production: Arr,
    arr_eta: Arr,
    arr_owner: Arr,
    arr_ships: Arr,
    horizon: int,
) -> tuple[Arr, Arr]:
    """Fold turns 1..horizon. Returns (final_owner, min_owned_garrison_for_player).

    Used both for the descriptive timeline and (with a custom keep) for the
    survival check. Mirrors the per-turn order: add production (if owned), then
    resolve arrivals.
    """

    def step(carry: tuple[Arr, Arr], turn: Arr) -> tuple[tuple[Arr, Arr], Arr]:
        owner, garrison = carry
        garrison = jnp.where(owner != -1, garrison + production, garrison)
        by_owner = _per_turn_owner_ships(arr_eta, arr_owner, arr_ships, turn)
        new_owner, new_g = resolve_arrival_event(owner, garrison, by_owner)
        return (new_owner, new_g), new_owner

    turns = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    (fowner, fgarrison), _owners = jax.lax.scan(step, (owner0, garrison0), turns)
    return fowner, fgarrison


def survives_with_keep(
    planet_owner: Arr,
    production: Arr,
    player: Arr,
    keep: Arr,
    arr_eta: Arr,
    arr_owner: Arr,
    arr_ships: Arr,
    horizon: int,
) -> Arr:
    """True iff starting with `keep` ships the planet stays `player`-owned.

    Exactly mirrors world_model.py:194-207: production added each turn (owner
    never -1 in the sim since it starts as player and we only check player
    survival), arrivals resolved; the moment owner != player after an arrival →
    fail. We track an `alive` flag through the scan to match the early-return.
    """

    def step(
        carry: tuple[Arr, Arr, Arr], turn: Arr
    ) -> tuple[tuple[Arr, Arr, Arr], None]:
        owner, garrison, alive = carry
        garrison = jnp.where(owner != -1, garrison + production, garrison)
        by_owner = _per_turn_owner_ships(arr_eta, arr_owner, arr_ships, turn)
        has_event = jnp.sum(by_owner) > 0
        new_owner, new_g = resolve_arrival_event(owner, garrison, by_owner)
        # Python: if group and sim_owner != player after resolve → return False
        failed_now = has_event & (new_owner != player)
        new_alive = alive & ~failed_now
        return (new_owner, new_g, new_alive), None

    turns = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    (fowner, _fg, alive), _ = jax.lax.scan(
        step, (planet_owner, keep.astype(jnp.float_), jnp.asarray(True)), turns
    )
    return alive & (fowner == player)


def keep_needed(
    planet_owner: Arr,
    planet_ships: Arr,
    production: Arr,
    player: Arr,
    arr_eta: Arr,
    arr_owner: Arr,
    arr_ships: Arr,
    horizon: int,
    max_ships: int,
) -> Arr:
    """Minimum keep in [0, ships] that survives; ships if even full doesn't hold.

    Mirrors the binary search (world_model.py:209-220) by evaluating all keep
    candidates 0..max_ships in parallel and taking the smallest that survives
    (only meaningful for keep <= ships). If full ships doesn't survive →
    keep_needed = ships (holds_full False in Python).
    """
    candidates = jnp.arange(0, max_ships + 1, dtype=jnp.int32)

    def check(keep: Arr) -> Arr:
        return survives_with_keep(
            planet_owner,
            production,
            player,
            keep,
            arr_eta,
            arr_owner,
            arr_ships,
            horizon,
        )

    surv = jax.vmap(check)(candidates)  # [max_ships+1] bool
    # only candidates <= planet_ships are valid keeps
    valid = candidates <= planet_ships
    # full_holds = survives_with_keep(ships); clamp index to candidate range.
    full_idx = jnp.clip(planet_ships, 0, max_ships)
    full_holds = surv[full_idx]
    # smallest surviving candidate among valid; if none, fall back to ships
    eligible = surv & valid
    big = max_ships + 1
    first = jnp.min(jnp.where(eligible, candidates, big))
    return jnp.where(full_holds, first, planet_ships).astype(jnp.int32)


def compute_reserve_per_planet(
    planet_owner: Arr,
    planet_ships: Arr,
    production: Arr,
    player: Arr,
    ledger_slot: Arr,
    ledger_eta: Arr,
    ledger_owner: Arr,
    ledger_ships: Arr,
    planet_slot: Arr,
    horizon: int,
    max_ships: int,
) -> Arr:
    """keep_needed reserve for one planet, with a no-arrivals short-circuit.

    ledger_* are per-fleet arrays. Only fleets whose target == planet_slot count
    as arrivals (any owner; the timeline resolves combat). If none → reserve 0
    (matches PoC0). Owner clamped to [0, NUM_PLAYERS) for the one-hot.
    """
    targets_me = (ledger_slot == planet_slot) & (ledger_eta > 0)
    enemy_threat = targets_me & (ledger_owner != player)
    has_threat = jnp.any(enemy_threat)
    arr_eta = jnp.where(targets_me, ledger_eta, 10**9)
    arr_owner = jnp.clip(ledger_owner, 0, NUM_PLAYERS - 1)
    arr_ships = jnp.where(targets_me, ledger_ships, 0.0)
    return jnp.where(
        has_threat,
        keep_needed(
            planet_owner,
            planet_ships,
            production,
            player,
            arr_eta,
            arr_owner,
            arr_ships,
            horizon,
            max_ships,
        ),
        0,
    ).astype(jnp.int32)
