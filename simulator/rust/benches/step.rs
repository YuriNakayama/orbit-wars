//! Criterion benchmark for the per-step interpreter.
//!
//! This benchmark exercises the pure-Rust step path with a representative
//! mid-game state (~10 planets, ~30 fleets) so the numbers are comparable
//! across iterations. It deliberately bypasses PyO3 — that boundary cost
//! is measured separately by the Python-side benchmark in
//! `python/tests/test_benchmark.py`.
//!
//! Usage:
//!   (cd simulator/rust && cargo bench --bench step)
//!
//! Each Phase of the speedup plan should re-run this and update
//! `docs/plans/rust-simulator/benchmark_results.md` with the new µs/iter.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use orbit_wars_rust::interpreter::step;
use orbit_wars_rust::state::{Fleet, Move, OrbitWarsState, Planet};

fn make_state(num_planets: usize, num_fleets: usize) -> OrbitWarsState {
    let mut state = OrbitWarsState::new(2, 0);
    state.angular_velocity = 0.03;
    state.step = 100;

    // Lay planets on a coarse grid inside the rotation band.
    for i in 0..num_planets {
        let angle = (i as f64) * 0.6;
        let r = 25.0 + ((i % 3) as f64) * 8.0;
        let x = 50.0 + r * angle.cos();
        let y = 50.0 + r * angle.sin();
        let owner = (i % 2) as i32;
        state.planets.push(Planet {
            id: i as i64,
            owner,
            x,
            y,
            radius: 1.5,
            ships: 25,
            production: 2,
        });
    }
    state.initial_planets = state.planets.clone();

    // Sprinkle fleets at random angles around the board.
    for i in 0..num_fleets {
        let angle = (i as f64) * 0.37;
        let from = (i % num_planets) as i64;
        let p = &state.planets[from as usize];
        state.fleets.push(Fleet {
            id: i as i64,
            owner: ((i % 2) as i32),
            x: p.x + 0.5 * angle.cos(),
            y: p.y + 0.5 * angle.sin(),
            angle,
            from_planet_id: from,
            ships: 5,
        });
    }
    state.next_fleet_id = num_fleets as i64;
    state
}

fn bench_step_noop(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_noop");
    for &(np, nf) in &[(10usize, 10usize), (24, 30), (32, 80)] {
        let id = BenchmarkId::from_parameter(format!("p{}_f{}", np, nf));
        group.bench_function(id, |b| {
            b.iter_batched(
                || make_state(np, nf),
                |mut st| {
                    step(&mut st, black_box(&[Vec::new(), Vec::new()]));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_step_with_moves(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_with_moves");
    let actions = vec![
        vec![Move {
            from_planet_id: 0,
            angle: 0.7,
            ships: 10,
        }],
        vec![Move {
            from_planet_id: 1,
            angle: 1.3,
            ships: 10,
        }],
    ];
    let id = BenchmarkId::from_parameter("p24_f30");
    group.bench_function(id, |b| {
        b.iter_batched(
            || make_state(24, 30),
            |mut st| {
                step(&mut st, black_box(&actions));
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_step_noop, bench_step_with_moves);
criterion_main!(benches);
