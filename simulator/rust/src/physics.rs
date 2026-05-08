//! Per-turn motion: fleet advance, planet rotation, comet path advance,
//! sweep collision detection.
//!
//! Mirrors the math in `simulator/python/orbit_wars_vendor/orbit_wars.py` lines
//! 30–43 (distance / point_to_segment_distance) and 519–626 (movement loops).
//! Reordering the phases is intentionally avoided — the upstream order
//! (process_moves → production → fleet move → planet move → comet → combat)
//! is part of the contract that parity tests pin down.

use crate::state::{OrbitWarsState, BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS};

/// Euclidean distance between two `(x, y)` points.
pub fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    (dx * dx + dy * dy).sqrt()
}

/// Squared Euclidean distance — useful when only `d < r` comparisons are
/// needed. Avoids the `sqrt` on the hot collision path (called once per
/// fleet × planet × turn = ~thousands per episode).
#[inline]
fn distance_sq(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    dx * dx + dy * dy
}

/// Minimum distance from point `p` to the line segment `v -> w`.
///
/// Verbatim port of upstream `point_to_segment_distance`. Used by both
/// fleet-vs-planet collision and planet-sweeps-fleet detection.
pub fn point_to_segment_distance(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    point_to_segment_distance_sq(p, v, w).sqrt()
}

/// Squared variant of `point_to_segment_distance`. Use this on hot paths
/// where only `d < r` comparisons are needed: substitute `d² < r²`. The
/// `sqrt` is preserved when callers actually need the linear distance.
///
/// **Bit-exact equivalence**: for any non-negative `r`, `d < r` is true
/// iff `d² < r²`. We compute `d²` via the same `(dx*dx + dy*dy)` formula
/// upstream uses, so the comparison result is identical to upstream's
/// `distance(...) < r` outcome.
#[inline]
pub fn point_to_segment_distance_sq(p: (f64, f64), v: (f64, f64), w: (f64, f64)) -> f64 {
    let l2 = (v.0 - w.0).powi(2) + (v.1 - w.1).powi(2);
    if l2 == 0.0 {
        return distance_sq(p, v);
    }
    let t = (((p.0 - v.0) * (w.0 - v.0) + (p.1 - v.1) * (w.1 - v.1)) / l2).clamp(0.0, 1.0);
    let projection = (v.0 + t * (w.0 - v.0), v.1 + t * (w.1 - v.1));
    distance_sq(p, projection)
}

/// True iff a fleet moving `a -> b` and a planet moving `p0 -> p1`
/// come within `radius` during the same tick.
pub fn swept_pair_hit(
    a: (f64, f64),
    b: (f64, f64),
    p0: (f64, f64),
    p1: (f64, f64),
    radius: f64,
) -> bool {
    let d0x = a.0 - p0.0;
    let d0y = a.1 - p0.1;
    let dvx = (b.0 - a.0) - (p1.0 - p0.0);
    let dvy = (b.1 - a.1) - (p1.1 - p0.1);
    let qa = dvx * dvx + dvy * dvy;
    let qb = 2.0 * (d0x * dvx + d0y * dvy);
    let qc = d0x * d0x + d0y * d0y - radius * radius;
    if qa < 1e-12 {
        return qc <= 0.0;
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    let t1 = (-qb - sq) / (2.0 * qa);
    let t2 = (-qb + sq) / (2.0 * qa);
    t2 >= 0.0 && t1 <= 1.0
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetPath {
    pub old_pos: (f64, f64),
    pub new_pos: (f64, f64),
    pub check_collision: bool,
}

/// Speed of a fleet of `ships` at `max_speed` (= `configuration.shipSpeed`).
///
/// Upstream formula (orbit_wars.py:528):
///   speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ** 1.5
///   speed = min(speed, max_speed)
///
/// Larger fleets approach the cap; a single ship moves at 1.0 / turn.
pub fn fleet_speed(ships: i64, max_speed: f64) -> f64 {
    let ratio = (ships as f64).ln() / 1000_f64.ln();
    let raw = 1.0 + (max_speed - 1.0) * ratio.powf(1.5);
    raw.min(max_speed)
}

/// Outcome of `advance_fleets` per fleet: either it moved (and its index in
/// the fleet list maps to the destination point) or it was eliminated. The
/// return surface is intentionally minimal so the caller can reuse the
/// per-planet collision lists when resolving combat.
#[derive(Debug, Default)]
pub struct FleetMovementOutcome {
    /// Per-planet `(planet_id, fleet_indices_intersected)` lists. Ordering
    /// matches `state.planets` ordering at construction time AND is preserved
    /// through every motion phase (planets are not added or removed during
    /// a single turn until `finalize_motion`).
    pub combat_lists: Vec<(i64, Vec<usize>)>,
    /// Per-fleet "marked for removal" flag, indexed by fleet index. Replaces
    /// HashSet — for the typical fleet count this is cache-friendly and
    /// avoids hash overhead.
    pub fleets_to_remove: Vec<bool>,
}

/// Phase 2 of the upstream interpreter: advance every fleet by its speed,
/// then test for sun crossing / out-of-bounds / planet hits.
///
/// Mutates `state.fleets` in place (positions advance) and returns the
/// metadata needed by combat / planet sweep.
pub fn compute_planet_paths(state: &mut OrbitWarsState) -> (Vec<PlanetPath>, Vec<i64>) {
    let mut paths: Vec<PlanetPath> = state
        .planets
        .iter()
        .map(|p| PlanetPath {
            old_pos: p.pos(),
            new_pos: p.pos(),
            check_collision: true,
        })
        .collect();
    let mut expired: Vec<i64> = Vec::new();

    let max_id = state
        .planets
        .iter()
        .map(|p| p.id)
        .chain(state.initial_planets.iter().map(|p| p.id))
        .max()
        .unwrap_or(-1);
    if max_id < 0 {
        return (paths, expired);
    }
    let mut slot_of = vec![-1i32; (max_id as usize) + 1];
    for (slot, p) in state.planets.iter().enumerate() {
        if p.id >= 0 && (p.id as usize) < slot_of.len() {
            slot_of[p.id as usize] = slot as i32;
        }
    }
    let mut is_comet = vec![false; slot_of.len()];
    for &cid in state.comet_planet_ids.iter() {
        if cid >= 0 && (cid as usize) < is_comet.len() {
            is_comet[cid as usize] = true;
        }
    }
    let mut initial_slot = vec![-1i32; slot_of.len()];
    for (i, p) in state.initial_planets.iter().enumerate() {
        if p.id >= 0 && (p.id as usize) < initial_slot.len() {
            initial_slot[p.id as usize] = i as i32;
        }
    }

    for slot in 0..state.planets.len() {
        let planet_id = state.planets[slot].id;
        if planet_id < 0 || (planet_id as usize) >= is_comet.len() || is_comet[planet_id as usize] {
            continue;
        }
        let init_idx = initial_slot[planet_id as usize];
        if init_idx < 0 {
            continue;
        }
        let initial = &state.initial_planets[init_idx as usize];
        let dx = initial.x - CENTER;
        let dy = initial.y - CENTER;
        let orbital_r = (dx * dx + dy * dy).sqrt();
        let radius = state.planets[slot].radius;
        if orbital_r + radius < ROTATION_RADIUS_LIMIT {
            let current_angle = dy.atan2(dx) + state.angular_velocity * state.step as f64;
            let (sin_c, cos_c) = current_angle.sin_cos();
            paths[slot].new_pos = (CENTER + orbital_r * cos_c, CENTER + orbital_r * sin_c);
        }
    }

    for group_idx in 0..state.comets.len() {
        state.comets[group_idx].path_index += 1;
        let path_idx = state.comets[group_idx].path_index;
        let planet_ids = state.comets[group_idx].planet_ids.clone();
        for (i, pid) in planet_ids.iter().enumerate() {
            if *pid < 0 || (*pid as usize) >= slot_of.len() {
                continue;
            }
            let slot_i = slot_of[*pid as usize];
            if slot_i < 0 {
                continue;
            }
            let slot = slot_i as usize;
            let old_pos = state.planets[slot].pos();
            let path = &state.comets[group_idx].paths[i];
            if (path_idx as usize) >= path.len() {
                expired.push(*pid);
                paths[slot] = PlanetPath {
                    old_pos,
                    new_pos: old_pos,
                    check_collision: true,
                };
                continue;
            }
            let target = path[path_idx as usize];
            paths[slot] = PlanetPath {
                old_pos,
                new_pos: (target[0], target[1]),
                check_collision: old_pos.0 >= 0.0,
            };
        }
    }

    (paths, expired)
}

/// Phase 3 of the upstream interpreter: advance every fleet by its speed,
/// then test swept-pair planet hits before bounds and sun checks.
pub fn advance_fleets(
    state: &mut OrbitWarsState,
    planet_paths: &[PlanetPath],
) -> FleetMovementOutcome {
    let combat_lists: Vec<(i64, Vec<usize>)> =
        state.planets.iter().map(|p| (p.id, Vec::new())).collect();
    let fleets_to_remove = vec![false; state.fleets.len()];
    let mut outcome = FleetMovementOutcome {
        combat_lists,
        fleets_to_remove,
    };

    for (idx, fleet) in state.fleets.iter_mut().enumerate() {
        let speed = fleet_speed(fleet.ships, state.ship_speed);
        let old_pos = (fleet.x, fleet.y);
        let (sin_a, cos_a) = fleet.angle.sin_cos();
        fleet.x += cos_a * speed;
        fleet.y += sin_a * speed;
        let new_pos = (fleet.x, fleet.y);

        let mut hit_planet = false;
        for (slot, planet) in state.planets.iter().enumerate() {
            let Some(path) = planet_paths.get(slot) else {
                continue;
            };
            if !path.check_collision {
                continue;
            }
            if swept_pair_hit(old_pos, new_pos, path.old_pos, path.new_pos, planet.radius) {
                outcome.combat_lists[slot].1.push(idx);
                outcome.fleets_to_remove[idx] = true;
                hit_planet = true;
                break;
            }
        }
        if hit_planet {
            continue;
        }

        if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
            outcome.fleets_to_remove[idx] = true;
            continue;
        }

        const SUN_RADIUS_SQ: f64 = SUN_RADIUS * SUN_RADIUS;
        if point_to_segment_distance_sq((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS_SQ {
            outcome.fleets_to_remove[idx] = true;
        }
    }

    outcome
}

pub fn apply_planet_paths(state: &mut OrbitWarsState, planet_paths: &[PlanetPath]) {
    for (planet, path) in state.planets.iter_mut().zip(planet_paths.iter()) {
        planet.x = path.new_pos.0;
        planet.y = path.new_pos.1;
    }
}

/// Rotate non-comet planets around the center, sweeping any fleet that lies
/// on the path. Adds the swept fleets to `outcome.combat_lists` /
/// `fleets_to_remove`.
pub fn rotate_planets(state: &mut OrbitWarsState, outcome: &mut FleetMovementOutcome) {
    let step = state.step;
    let av = state.angular_velocity;

    // Build two id-indexed tables once per call: a comet bitmap and an
    // initial_planets slot lookup. Both replace `Vec::contains` /
    // `Iterator::find` linear scans inside the per-planet loop. Sized to
    // `max_id + 1` so each index is one bounds-checked read.
    let max_id = state
        .planets
        .iter()
        .map(|p| p.id)
        .chain(state.initial_planets.iter().map(|p| p.id))
        .max()
        .unwrap_or(-1);
    if max_id < 0 {
        return;
    }
    let table_len = max_id as usize + 1;
    let mut is_comet = vec![false; table_len];
    for &cid in state.comet_planet_ids.iter() {
        if cid >= 0 && (cid as usize) < table_len {
            is_comet[cid as usize] = true;
        }
    }
    let mut initial_slot = vec![-1i32; table_len];
    for (i, p) in state.initial_planets.iter().enumerate() {
        if p.id >= 0 && (p.id as usize) < table_len {
            initial_slot[p.id as usize] = i as i32;
        }
    }

    for slot in 0..state.planets.len() {
        let planet_id = state.planets[slot].id;
        if planet_id < 0 || (planet_id as usize) >= table_len {
            continue;
        }
        if is_comet[planet_id as usize] {
            continue;
        }
        let init_idx = initial_slot[planet_id as usize];
        if init_idx < 0 {
            continue;
        }
        let initial = &state.initial_planets[init_idx as usize];
        let ix = initial.x;
        let iy = initial.y;
        let dx = ix - CENTER;
        let dy = iy - CENTER;
        let r = (dx * dx + dy * dy).sqrt();
        let radius = state.planets[slot].radius;
        let old_pos = state.planets[slot].pos();

        if r + radius < ROTATION_RADIUS_LIMIT {
            let initial_angle = dy.atan2(dx);
            let current_angle = initial_angle + av * step as f64;
            let (sin_c, cos_c) = current_angle.sin_cos();
            let new_x = CENTER + r * cos_c;
            let new_y = CENTER + r * sin_c;
            let planet_mut = &mut state.planets[slot];
            planet_mut.x = new_x;
            planet_mut.y = new_y;
        }

        let new_pos = state.planets[slot].pos();
        sweep_fleets_on_planet(state, slot, old_pos, new_pos, outcome);
    }
}

/// Advance comet groups along their pre-computed paths, sweeping fleets along
/// the way. Returns the comet planet ids that have walked off the end of
/// their path (caller is expected to delete them).
pub fn advance_comets(state: &mut OrbitWarsState, outcome: &mut FleetMovementOutcome) -> Vec<i64> {
    let mut expired: Vec<i64> = Vec::new();

    // planet_id → slot table, built once per call instead of one
    // `iter().position(...)` per (group × member) lookup.
    let max_id = state.planets.iter().map(|p| p.id).max().unwrap_or(-1);
    if max_id < 0 {
        return expired;
    }
    let mut slot_of = vec![-1i32; (max_id as usize) + 1];
    for (i, p) in state.planets.iter().enumerate() {
        if p.id >= 0 && (p.id as usize) < slot_of.len() {
            slot_of[p.id as usize] = i as i32;
        }
    }

    for group_idx in 0..state.comets.len() {
        state.comets[group_idx].path_index += 1;
        let path_idx = state.comets[group_idx].path_index;

        // Snapshot the planet ids and paths to avoid &mut/&immut overlap.
        let planet_ids = state.comets[group_idx].planet_ids.clone();
        for (i, pid) in planet_ids.iter().enumerate() {
            if *pid < 0 || (*pid as usize) >= slot_of.len() {
                continue;
            }
            let slot_i = slot_of[*pid as usize];
            if slot_i < 0 {
                continue;
            }
            let slot = slot_i as usize;
            let path = &state.comets[group_idx].paths[i];
            if (path_idx as usize) >= path.len() {
                expired.push(*pid);
                continue;
            }
            let target = path[path_idx as usize];
            let old_pos = state.planets[slot].pos();
            state.planets[slot].x = target[0];
            state.planets[slot].y = target[1];
            // Skip sweep on first placement (old_pos is the off-board placeholder
            // -99,-99 used by spawn_comets).
            if old_pos.0 >= 0.0 {
                let new_pos = state.planets[slot].pos();
                sweep_fleets_on_planet(state, slot, old_pos, new_pos, outcome);
            }
        }
    }
    expired
}

/// Helper: register any fleet whose centre lies within `planet.radius` of the
/// segment `old_pos -> new_pos` as caught by the planet at `slot`.
fn sweep_fleets_on_planet(
    state: &OrbitWarsState,
    slot: usize,
    old_pos: (f64, f64),
    new_pos: (f64, f64),
    outcome: &mut FleetMovementOutcome,
) {
    if old_pos == new_pos {
        return;
    }
    let planet = &state.planets[slot];
    let radius = planet.radius;
    let radius_sq = radius * radius;

    // Bounding box around the planet's swept segment, padded by radius.
    // Fleets outside this box can't possibly be within `radius` of the
    // segment, so we can reject with a single coordinate compare per axis
    // before touching the f64 multiply chain.
    let min_x = old_pos.0.min(new_pos.0) - radius;
    let max_x = old_pos.0.max(new_pos.0) + radius;
    let min_y = old_pos.1.min(new_pos.1) - radius;
    let max_y = old_pos.1.max(new_pos.1) + radius;

    // `combat_lists[slot]` corresponds to `state.planets[slot]` because both
    // are built in the same order in `advance_fleets`, and motion phases do
    // not add/remove planets within a turn (finalize_motion runs later).
    for (idx, fleet) in state.fleets.iter().enumerate() {
        if outcome.fleets_to_remove[idx] {
            continue;
        }
        let fx = fleet.x;
        let fy = fleet.y;
        if fx < min_x || fx > max_x || fy < min_y || fy > max_y {
            continue;
        }
        if point_to_segment_distance_sq((fx, fy), old_pos, new_pos) < radius_sq {
            outcome.combat_lists[slot].1.push(idx);
            outcome.fleets_to_remove[idx] = true;
        }
    }
}

/// After all motion phases, prune fleets marked for removal and apply the
/// pending comet expirations to `planets` / `initial_planets` / `comets`.
pub fn finalize_motion(
    state: &mut OrbitWarsState,
    outcome: &FleetMovementOutcome,
    expired_comet_pids: &[i64],
) {
    // `fleets_to_remove` is a Vec<bool> the same length as `state.fleets`
    // at outcome construction. Run the retain only when at least one flag
    // is set; `iter().any` short-circuits on the first true.
    if outcome.fleets_to_remove.iter().any(|&b| b) {
        let to_remove = &outcome.fleets_to_remove;
        let mut idx = 0usize;
        state.fleets.retain(|_| {
            // Direct index — vec is sized exactly to state.fleets.len() at
            // outcome construction, so out-of-range is impossible.
            let keep = !to_remove[idx];
            idx += 1;
            keep
        });
    }
    if !expired_comet_pids.is_empty() {
        // Expired ids are at most 4 (one comet group), so linear scan beats
        // a HashSet allocation + hashing.
        let expired = expired_comet_pids;
        state.planets.retain(|p| !expired.contains(&p.id));
        state.initial_planets.retain(|p| !expired.contains(&p.id));
        state.comet_planet_ids.retain(|pid| !expired.contains(pid));
        for group in state.comets.iter_mut() {
            group.planet_ids.retain(|pid| !expired.contains(pid));
        }
        state.comets.retain(|g| !g.planet_ids.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_basic() {
        assert!((distance((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn point_to_segment_zero_length() {
        // Degenerate segment falls back to point distance.
        assert!(
            (point_to_segment_distance((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)) - 2.0_f64.sqrt()).abs()
                < 1e-9
        );
    }

    #[test]
    fn point_to_segment_perpendicular() {
        // Closest point on the x-axis segment to (0, 5) is the origin.
        let d = point_to_segment_distance((0.0, 5.0), (-10.0, 0.0), (10.0, 0.0));
        assert!((d - 5.0).abs() < 1e-9);
    }

    #[test]
    fn swept_pair_hit_detects_crossing_motion() {
        assert!(swept_pair_hit(
            (5.0, -5.0),
            (5.0, 5.0),
            (0.0, 0.0),
            (10.0, 0.0),
            1.0,
        ));
        assert!(!swept_pair_hit(
            (5.0, -5.0),
            (5.0, -3.0),
            (0.0, 0.0),
            (10.0, 0.0),
            1.0,
        ));
    }

    #[test]
    fn fleet_speed_one_ship_is_one() {
        // log(1) = 0 → speed exactly 1.0 regardless of cap.
        assert!((fleet_speed(1, 6.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fleet_speed_caps_at_max() {
        // Very large fleet asymptotes to (and is clamped at) max_speed.
        let s = fleet_speed(10_000, 6.0);
        assert!(s <= 6.0 + 1e-9);
    }
}
