//! Game state types — `Planet`, `Fleet`, `CometGroup`, `OrbitWarsState`.
//!
//! Layouts mirror the upstream `[id, owner, x, y, radius, ships, production]`
//! tuples used by `kaggle_environments.envs.orbit_wars`. We keep them as
//! struct-of-fields here because Rust pattern-matching against tuple indices
//! gets ugly fast; the conversion back into the Python list/dict shape is
//! `pybind.rs`'s job.
//!
//! The Python interpreter mutates a shared `state[0].observation` namespace.
//! We instead own a single `OrbitWarsState` and rebuild the observation dict
//! in place — see `pybind::write_state_back`.

use crate::rng::ChaChaRng;

/// Board / engine constants. Keep in sync with the upstream `orbit_wars.py`.
pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = BOARD_SIZE / 2.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const PLANET_CLEARANCE: f64 = 7.0;
pub const MIN_PLANET_GROUPS: i64 = 5;
pub const MAX_PLANET_GROUPS: i64 = 10;
pub const MIN_STATIC_GROUPS: i64 = 3;
pub const COMET_SPAWN_STEPS: [i64; 5] = [50, 150, 250, 350, 450];

/// Per-planet record. Mirrors `[id, owner, x, y, radius, ships, production]`.
///
/// `owner == -1` is the neutral / unowned sentinel (see upstream constants).
#[derive(Clone, Debug, PartialEq)]
pub struct Planet {
    pub id: i64,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

impl Planet {
    pub fn pos(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

/// Per-fleet record. Mirrors `[id, owner, x, y, angle, from_planet_id, ships]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Fleet {
    pub id: i64,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

impl Fleet {
    pub fn pos(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

/// Group of 4 symmetric comets that share `paths` and an advance index.
#[derive(Clone, Debug, PartialEq)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

/// Single per-player action: `[from_planet_id, angle, ships]` (after sanitizing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub from_planet_id: i64,
    pub angle: f64,
    pub ships: i64,
}

/// Full simulation state. Owned by the Rust interpreter and rebuilt into the
/// Python observation dict at each `step` boundary.
#[derive(Clone, Debug)]
pub struct OrbitWarsState {
    /// Number of agents (2 or 4 per the upstream specification).
    pub num_agents: usize,
    /// Per-turn rotation rate sampled at game-start (~U[0.025, 0.05]).
    pub angular_velocity: f64,
    /// Live planets (regular + comets). Indexed by `id` via linear scan.
    pub planets: Vec<Planet>,
    /// Frozen snapshot of planets at game-start (used to compute the orbit
    /// reference angle for rotating planets).
    pub initial_planets: Vec<Planet>,
    /// In-flight fleets.
    pub fleets: Vec<Fleet>,
    /// Next free fleet id; assigned on launch.
    pub next_fleet_id: i64,
    /// Active comet groups (each spans 4 symmetric paths).
    pub comets: Vec<CometGroup>,
    /// Flat set of comet planet ids — denormalized for O(1) lookups.
    pub comet_planet_ids: Vec<i64>,
    /// Step count maintained by `Environment` and surfaced as `obs.step`.
    pub step: i64,
    /// Episode-length budget (mirrors `configuration.episodeSteps`).
    pub episode_steps: i64,
    /// Max fleet speed cap from `configuration.shipSpeed`.
    pub ship_speed: f64,
    /// Comet path speed from `configuration.cometSpeed`.
    pub comet_speed: f64,
    /// Per-player reward, populated only at termination.
    pub rewards: Vec<i32>,
    /// Whether the episode has reached a terminal turn.
    pub done: bool,
    /// Source of randomness for planet/comet generation.
    pub rng: ChaChaRng,
}

impl OrbitWarsState {
    /// Construct an "uninitialized" state. The interpreter populates planets
    /// and home-planet ownership on the first call (mirroring the upstream
    /// `if not obs0.planets` branch).
    pub fn new(num_agents: usize, seed: u64) -> Self {
        Self {
            num_agents,
            angular_velocity: 0.0,
            planets: Vec::new(),
            initial_planets: Vec::new(),
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
            step: 0,
            episode_steps: 500,
            ship_speed: 6.0,
            comet_speed: 4.0,
            rewards: vec![0; num_agents],
            done: false,
            rng: ChaChaRng::seeded(seed),
        }
    }

    pub fn planet_by_id(&self, id: i64) -> Option<&Planet> {
        self.planets.iter().find(|p| p.id == id)
    }

    pub fn planet_by_id_mut(&mut self, id: i64) -> Option<&mut Planet> {
        self.planets.iter_mut().find(|p| p.id == id)
    }

    pub fn initial_planet_by_id(&self, id: i64) -> Option<&Planet> {
        self.initial_planets.iter().find(|p| p.id == id)
    }
}
