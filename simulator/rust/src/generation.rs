//! Initial planet generation and comet path sampling.
//!
//! `generate_planets` is intentionally NOT ported (rejection sampling against
//! Python's global `random` module is hard to mirror cross-language). The
//! facade keeps that one-time bootstrap on the Python side.
//!
//! `generate_comet_paths`, however, is called 5 times per episode and is the
//! single most expensive Python step (~14 ms each, ~70 ms / episode total).
//! We port the math here but **do not** consume any RNG inside Rust — the
//! Python facade pre-generates up to 300 attempts worth of `(e, a, phi)`
//! uniform samples by calling `random.uniform` exactly as upstream does, then
//! hands them to Rust. This preserves the upstream's `random` consumption
//! order so initial planets / homeworld assignment / comet ship counts stay
//! bit-exact with the Python interpreter.
//!
//! Returns `None` (matching Python's `return None`) when all 300 attempts
//! fail. Otherwise returns 4 symmetric paths of [x, y] points.

use crate::physics::distance;
use crate::state::{
    Planet, BOARD_SIZE, CENTER, COMET_RADIUS, PLANET_CLEARANCE, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};

/// One pre-sampled rejection-sampling attempt: `e, a, phi` triple from
/// `random.uniform(0.75, 0.93) / random.uniform(60, 150) / random.uniform(π/6, π/3)`.
#[derive(Clone, Copy, Debug)]
pub struct CometAttempt {
    pub eccentricity: f64,
    pub semi_major: f64,
    pub orientation: f64,
}

/// Phase 1 (static planet groups) attempt: `prod`, `angle`, `orbital_r`,
/// `ships`. Pre-sampled from upstream order:
///   prod = randint(1, 5)
///   angle = uniform(0, π/2)
///   if max_orbital < min_orbital: continue   # consume nothing more
///   orbital_r = uniform(min_orbital, max_orbital)
///   if board-edge / axis-clearance fails: continue   # consume nothing more
///   ships = min(randint(5,99), randint(5,99))
///
/// Bool flags carry the dispatch info.
#[derive(Clone, Copy, Debug)]
pub struct PlanetPhase1Attempt {
    pub prod: i64,
    pub angle: f64,
    pub orbital_r: f64,
    pub ships: i64,
    /// `min_orbital > max_orbital` の早期 skip フラグ。Rust 側ではこの場合
    /// orbital_r/ships は意味なし。
    pub orbital_skip: bool,
}

/// Phase 1.5 (diagonal orbiting group) attempt:
///   prod = randint(1, 5)
///   if min_orbital >= max_orbital: continue   # consume nothing more
///   orbital_r = uniform(min_orbital, max_orbital)
///   ships = min(randint(5,99), randint(5,99))
#[derive(Clone, Copy, Debug)]
pub struct PlanetPhase15Attempt {
    pub prod: i64,
    pub orbital_r: f64,
    pub ships: i64,
    pub orbital_skip: bool,
}

/// Phase 2 (fill loop) attempt:
///   prod = randint(1, 5)
///   x = uniform(CENTER + 15, BOARD_SIZE - r - 5)
///   y = uniform(CENTER + 15, BOARD_SIZE - r - 5)
///   ships = randint(5, 30)
#[derive(Clone, Copy, Debug)]
pub struct PlanetPhase2Attempt {
    pub prod: i64,
    pub x: f64,
    pub y: f64,
    pub ships: i64,
}

/// Run all 3 phases of upstream `generate_planets` against pre-sampled
/// random attempts. Returns the planets vec (length is multiple of 4).
///
/// `num_q1` is `random.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)` drawn
/// by the caller before phase 1.
pub fn generate_planets(
    num_q1: i64,
    phase1: &[PlanetPhase1Attempt],
    phase15: &[PlanetPhase15Attempt],
    phase2: &[PlanetPhase2Attempt],
) -> Vec<Planet> {
    let mut planets: Vec<Planet> = Vec::new();
    let mut id_counter: i64 = 0;

    // Phase 1: 3 guaranteed static planet groups
    let mut static_groups = 0i64;
    for att in phase1.iter() {
        if static_groups >= 3 {
            break;
        }
        if att.orbital_skip {
            continue;
        }
        let r = 1.0 + (att.prod as f64).ln();
        let x = CENTER + att.orbital_r * att.angle.cos();
        let y = CENTER + att.orbital_r * att.angle.sin();

        if x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0 {
            continue;
        }
        if (BOARD_SIZE - x) - r < 0.0 || (BOARD_SIZE - y) - r < 0.0 {
            continue;
        }
        if (x - CENTER) < r + 5.0 || (y - CENTER) < r + 5.0 {
            continue;
        }

        let temp = mirror4(id_counter, x, y, r, att.ships, att.prod);
        if !overlaps_existing_simple(&planets, &temp) {
            planets.extend(temp);
            id_counter += 4;
            static_groups += 1;
        }
    }

    let _ = phase15;

    // Phase 2: fill remaining
    let mut has_orbiting = false;
    for att in phase2.iter() {
        let limit_groups = (num_q1 as usize) * 4;
        if planets.len() >= limit_groups && has_orbiting {
            break;
        }
        let r = 1.0 + (att.prod as f64).ln();
        let x = att.x;
        let y = att.y;
        let orbital_radius = distance((x, y), (CENTER, CENTER));
        if orbital_radius < SUN_RADIUS + r + 10.0 {
            continue;
        }
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT
            && (x + r > BOARD_SIZE || x - r < 0.0 || y + r > BOARD_SIZE || y - r < 0.0)
        {
            continue;
        }
        let temp = mirror4(id_counter, x, y, r, att.ships, att.prod);
        if !overlaps_with_rotation_classification(&planets, &temp) {
            if orbital_radius + r < ROTATION_RADIUS_LIMIT {
                has_orbiting = true;
            }
            planets.extend(temp);
            id_counter += 4;
        }
    }

    planets
}

fn mirror4(id0: i64, x: f64, y: f64, r: f64, ships: i64, prod: i64) -> Vec<Planet> {
    vec![
        Planet {
            id: id0,
            owner: -1,
            x: y,
            y: x,
            radius: r,
            ships,
            production: prod,
        },
        Planet {
            id: id0 + 1,
            owner: -1,
            x: BOARD_SIZE - x,
            y,
            radius: r,
            ships,
            production: prod,
        },
        Planet {
            id: id0 + 2,
            owner: -1,
            x,
            y: BOARD_SIZE - y,
            radius: r,
            ships,
            production: prod,
        },
        Planet {
            id: id0 + 3,
            owner: -1,
            x: BOARD_SIZE - y,
            y: BOARD_SIZE - x,
            radius: r,
            ships,
            production: prod,
        },
    ]
}

fn overlaps_existing_simple(existing: &[Planet], temp: &[Planet]) -> bool {
    for tp in temp {
        for p in existing {
            if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                return true;
            }
        }
    }
    false
}

fn overlaps_with_orbital_check(existing: &[Planet], temp: &[Planet]) -> bool {
    for tp in temp {
        let tp_orbital = distance((tp.x, tp.y), (CENTER, CENTER));
        for p in existing {
            let p_orbital = distance((p.x, p.y), (CENTER, CENTER));
            let p_is_static = p_orbital + p.radius >= ROTATION_RADIUS_LIMIT;
            if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                return true;
            }
            if p_is_static
                && (tp_orbital - p_orbital).abs() < tp.radius + p.radius + PLANET_CLEARANCE
            {
                return true;
            }
        }
    }
    false
}

fn overlaps_with_rotation_classification(existing: &[Planet], temp: &[Planet]) -> bool {
    for tp in temp {
        let tp_orbital = distance((tp.x, tp.y), (CENTER, CENTER));
        let tp_is_rotating = tp_orbital + tp.radius < ROTATION_RADIUS_LIMIT;
        for p in existing {
            let p_orbital = distance((p.x, p.y), (CENTER, CENTER));
            let p_is_rotating = p_orbital + p.radius < ROTATION_RADIUS_LIMIT;
            if distance((p.x, p.y), (tp.x, tp.y)) < p.radius + tp.radius + PLANET_CLEARANCE {
                return true;
            }
            if tp_is_rotating != p_is_rotating
                && (tp_orbital - p_orbital).abs() < tp.radius + p.radius + PLANET_CLEARANCE
            {
                return true;
            }
        }
    }
    false
}

/// Try one elliptical orbit and return the visible board segment (`(x, y)`
/// pairs) if all per-step collision checks pass. None if rejected.
fn try_attempt(
    attempt: CometAttempt,
    spawn_step: i64,
    angular_velocity: f64,
    comet_speed: f64,
    initial_planets: &[Planet],
    comet_planet_ids: &[i64],
) -> Option<Vec<(f64, f64)>> {
    let e = attempt.eccentricity;
    let a = attempt.semi_major;
    let phi = attempt.orientation;

    let perihelion = a * (1.0 - e);
    if perihelion < SUN_RADIUS + COMET_RADIUS {
        return None;
    }
    let b = a * (1.0 - e * e).sqrt();
    let c_val = a * e;

    // Dense sample around perihelion half of orbit.
    let num = 5000usize;
    let mut dense: Vec<(f64, f64)> = Vec::with_capacity(num);
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    for i in 0..num {
        let t =
            0.3 * std::f64::consts::PI + 1.4 * std::f64::consts::PI * i as f64 / (num as f64 - 1.0);
        let ex = c_val + a * t.cos();
        let ey = b * t.sin();
        let x = CENTER + ex * cos_phi - ey * sin_phi;
        let y = CENTER + ex * sin_phi + ey * cos_phi;
        dense.push((x, y));
    }

    // Re-sample at constant comet_speed arc-length intervals.
    let mut path: Vec<(f64, f64)> = Vec::new();
    path.push(dense[0]);
    let mut cum = 0.0;
    let mut target = comet_speed;
    for i in 1..dense.len() {
        cum += distance(dense[i], dense[i - 1]);
        if cum >= target {
            path.push(dense[i]);
            target += comet_speed;
        }
    }

    // Extract contiguous on-board segment.
    let mut board_start: Option<usize> = None;
    let mut board_end: Option<usize> = None;
    for (i, (x, y)) in path.iter().enumerate() {
        if (0.0..=BOARD_SIZE).contains(x) && (0.0..=BOARD_SIZE).contains(y) {
            if board_start.is_none() {
                board_start = Some(i);
            }
            board_end = Some(i);
        }
    }
    let (s, e_idx) = match (board_start, board_end) {
        (Some(s), Some(e_idx)) => (s, e_idx),
        _ => return None,
    };
    let visible = &path[s..=e_idx];
    if !(5..=40).contains(&visible.len()) {
        return None;
    }

    // Separate non-comet planets into static and orbiting.
    let comet_set: std::collections::HashSet<i64> = comet_planet_ids.iter().copied().collect();
    let mut static_planets: Vec<&Planet> = Vec::new();
    let mut orbiting_planets: Vec<&Planet> = Vec::new();
    for planet in initial_planets {
        if comet_set.contains(&planet.id) {
            continue;
        }
        let pr = distance((planet.x, planet.y), (CENTER, CENTER));
        if pr + planet.radius < ROTATION_RADIUS_LIMIT {
            orbiting_planets.push(planet);
        } else {
            static_planets.push(planet);
        }
    }

    let buf = COMET_RADIUS + 0.5;
    for (k, &(cx, cy)) in visible.iter().enumerate() {
        // Sun
        if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS {
            return None;
        }
        // 4 symmetric points
        let sym_pts = [
            (cy, cx),
            (BOARD_SIZE - cx, cy),
            (cx, BOARD_SIZE - cy),
            (BOARD_SIZE - cy, BOARD_SIZE - cx),
        ];
        // Static planets
        for planet in &static_planets {
            for sp in &sym_pts {
                if distance(*sp, (planet.x, planet.y)) < planet.radius + buf {
                    return None;
                }
            }
        }
        // Orbiting planets at their position at game step (spawn_step-1+k)
        let game_step = spawn_step - 1 + k as i64;
        for planet in &orbiting_planets {
            let dx = planet.x - CENTER;
            let dy = planet.y - CENTER;
            let orb_r = (dx * dx + dy * dy).sqrt();
            let init_angle = dy.atan2(dx);
            let cur_angle = init_angle + angular_velocity * game_step as f64;
            let px = CENTER + orb_r * cur_angle.cos();
            let py = CENTER + orb_r * cur_angle.sin();
            for sp in &sym_pts {
                if distance(*sp, (px, py)) < planet.radius + COMET_RADIUS {
                    return None;
                }
            }
        }
    }

    Some(visible.to_vec())
}

/// Try each pre-sampled `attempts` triple in order. Return the 4 symmetric
/// paths of the first successful attempt. None if all are rejected.
///
/// `spawn_step` is the game turn at which the comet spawn fires (matches
/// upstream `step + 1`).
pub fn generate_comet_paths(
    attempts: &[CometAttempt],
    spawn_step: i64,
    angular_velocity: f64,
    comet_speed: f64,
    initial_planets: &[Planet],
    comet_planet_ids: &[i64],
) -> Option<[Vec<[f64; 2]>; 4]> {
    for &attempt in attempts {
        if let Some(visible) = try_attempt(
            attempt,
            spawn_step,
            angular_velocity,
            comet_speed,
            initial_planets,
            comet_planet_ids,
        ) {
            // 4 rotationally symmetric paths.
            let p0: Vec<[f64; 2]> = visible.iter().map(|&(x, y)| [y, x]).collect();
            let p1: Vec<[f64; 2]> = visible.iter().map(|&(x, y)| [BOARD_SIZE - x, y]).collect();
            let p2: Vec<[f64; 2]> = visible.iter().map(|&(x, y)| [x, BOARD_SIZE - y]).collect();
            let p3: Vec<[f64; 2]> = visible
                .iter()
                .map(|&(x, y)| [BOARD_SIZE - y, BOARD_SIZE - x])
                .collect();
            return Some([p0, p1, p2, p3]);
        }
    }
    None
}
