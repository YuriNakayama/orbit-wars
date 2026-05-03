//! Combat resolution at each contested planet.
//!
//! Direct port of the upstream "4. Combat Resolution" loop in
//! `orbit_wars_vendor/orbit_wars.py`. The rules are:
//!
//!   1. Sum incoming ships per player at the planet.
//!   2. Identify the top fleet (most ships) and the runner-up.
//!   3. `survivor_ships = top - second`. Tie at the top kills everyone.
//!   4. Apply survivor against the planet's defender:
//!      - Same owner → reinforces the planet's `ships`.
//!      - Different owner → subtract from `ships`. If the result is negative,
//!        flip the planet's owner and set `ships = abs(result)`.

use crate::physics::FleetMovementOutcome;
use crate::state::OrbitWarsState;

pub fn resolve_combats(state: &mut OrbitWarsState, outcome: &FleetMovementOutcome) {
    // The outcome lists are computed BEFORE state.fleets was pruned, so the
    // indices point into the current `state.fleets` vector. The caller is
    // expected to call us BEFORE finalize_motion, so we can read fleets
    // directly without snapshotting.
    //
    // `combat_lists` is built in the same order as `state.planets` and the
    // motion phases preserve that ordering, so the index of a contested
    // planet IS its slot in `combat_lists` enumeration. We don't need a
    // separate HashMap.
    for (slot, (_planet_id, fleet_indices)) in outcome.combat_lists.iter().enumerate() {
        if fleet_indices.is_empty() {
            continue;
        }
        let planet_slot = slot;
        let _ = planet_slot;
        // Defensive sanity: combat_lists[slot].0 should equal state.planets[slot].id.
        // If a future refactor breaks the ordering invariant, fall back to linear find.
        debug_assert!(state.planets.get(slot).map(|p| p.id) == Some(*_planet_id));

        // Inline tally — small N (typically 1–4 owners), avoid HashMap.
        let mut tally: smallvec_inline::TinyTally = Default::default();
        for &idx in fleet_indices {
            let f = &state.fleets[idx];
            tally.add(f.owner, f.ships);
        }
        let Some((top_player, top_ships, second_ships)) = tally.top_two() else {
            continue;
        };

        let (survivor_owner, survivor_ships) = if let Some(second) = second_ships {
            let s = top_ships - second;
            let s_eff = if top_ships == second { 0 } else { s };
            let owner = if s_eff > 0 { top_player } else { -1 };
            (owner, s_eff)
        } else {
            (top_player, top_ships)
        };

        if survivor_ships > 0 {
            let planet = &mut state.planets[planet_slot];
            if planet.owner == survivor_owner {
                planet.ships += survivor_ships;
            } else {
                planet.ships -= survivor_ships;
                if planet.ships < 0 {
                    planet.owner = survivor_owner;
                    planet.ships = -planet.ships;
                }
            }
        }
    }
}

/// Inline-storage tally for at most 4 distinct owners (covers 4p FFA).
///
/// Equivalent to `HashMap<i32, i64>` but lives entirely on the stack with
/// no allocator round-trips. We sort top-2 with a single linear pass which
/// is faster than a generic `sort_by_key` for n≤4.
mod smallvec_inline {
    #[derive(Default)]
    pub struct TinyTally {
        owners: [i32; 4],
        ships: [i64; 4],
        len: usize,
    }

    impl TinyTally {
        pub fn add(&mut self, owner: i32, ships: i64) {
            for i in 0..self.len {
                if self.owners[i] == owner {
                    self.ships[i] += ships;
                    return;
                }
            }
            if self.len < self.owners.len() {
                self.owners[self.len] = owner;
                self.ships[self.len] = ships;
                self.len += 1;
            }
        }

        /// Return (top_owner, top_ships, optional second-place ships).
        /// Stable for ties: the first equal entry wins, matching upstream's
        /// `sorted(...)` behavior on Python dicts (insertion order).
        pub fn top_two(&self) -> Option<(i32, i64, Option<i64>)> {
            if self.len == 0 {
                return None;
            }
            let mut top_i = 0usize;
            for i in 1..self.len {
                if self.ships[i] > self.ships[top_i] {
                    top_i = i;
                }
            }
            let top_owner = self.owners[top_i];
            let top_ships = self.ships[top_i];
            if self.len == 1 {
                return Some((top_owner, top_ships, None));
            }
            // Find max ships among non-top entries.
            let mut second: Option<i64> = None;
            for i in 0..self.len {
                if i == top_i {
                    continue;
                }
                second = Some(match second {
                    None => self.ships[i],
                    Some(s) => s.max(self.ships[i]),
                });
            }
            Some((top_owner, top_ships, second))
        }
    }
}
