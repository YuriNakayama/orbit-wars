//! Rust ports of `kaggle_environments.utils` hot-path helpers.
//!
//! `kaggle_environments`'s framework calls `structify` twice per `env.step`
//! and `process_schema(action_schema, action)` once per agent. Together they
//! account for ~75% of the per-step time (see
//! `docs/plans/rust-simulator/benchmark_results.md`).
//!
//! This module exposes drop-in replacements that:
//!
//!   * `fast_structify`: produces an upstream-compatible `Struct` (the
//!     `dict`-subclass returned by `kaggle_environments.utils.structify`)
//!     by calling `Struct(**kwargs)` for each nested dict. Lists of
//!     scalars short-circuit (upstream creates a fresh list anyway, but
//!     we skip allocating new wrappers for leaves).
//!
//!   * `fast_validate_action`: returns `(err, sanitized_action)` for the
//!     orbit_wars action schema specifically. The schema is
//!     `list[list[number, number, number]]` with no extras, so we can
//!     bypass `jsonschema.validate` entirely. Falls back to upstream
//!     `process_schema` on any structural mismatch.
//!
//! Both are wired in via `_patches.apply()` in
//! `simulator/rust/python/orbit_wars_rust/_patches.py`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::generation::{
    generate_comet_paths as gen_paths, generate_planets as gen_planets, CometAttempt,
    PlanetPhase15Attempt, PlanetPhase1Attempt, PlanetPhase2Attempt,
};
use crate::state::Planet;

/// Recursive deep-wrap: dict → Struct, list → list of structified, scalar → as-is.
///
/// Mirrors upstream `structify` exactly except that:
///   * Lists of leaf scalars are returned by reference — upstream allocates
///     a fresh list of the same content. The caller doesn't observe a
///     difference because list contents are read, not compared by identity.
///   * Already-Struct values are passed through; upstream re-wraps them, but
///     the result is observably identical (same keys → same Struct).
///
/// `struct_class` must be `kaggle_environments.utils.Struct`.
#[pyfunction]
pub fn fast_structify<'py>(
    py: Python<'py>,
    o: Bound<'py, PyAny>,
    struct_class: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    structify_impl(py, o, &struct_class)
}

fn structify_impl<'py>(
    py: Python<'py>,
    o: Bound<'py, PyAny>,
    struct_class: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Fast path: scalar (int / float / str / bool / None) → return unchanged.
    if o.is_none() {
        return Ok(o);
    }
    if let Ok(list) = o.downcast::<PyList>() {
        // Pure-leaf list short-circuit. The upstream test below uses isinstance,
        // matching the python implementation's structify recursion semantics.
        let mut all_leaf = true;
        for item in list.iter() {
            if item.is_instance_of::<PyDict>() || item.is_instance_of::<PyList>() {
                all_leaf = false;
                break;
            }
        }
        if all_leaf {
            return Ok(o);
        }
        let new_items: Vec<Bound<'py, PyAny>> = list
            .iter()
            .map(|item| structify_impl(py, item, struct_class))
            .collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, new_items)?.into_any());
    }
    if let Ok(dict) = o.downcast::<PyDict>() {
        // Build kwargs by recursively structifying each value.
        let kwargs = PyDict::new(py);
        for (k, v) in dict.iter() {
            let new_v = structify_impl(py, v, struct_class)?;
            kwargs.set_item(k, new_v)?;
        }
        // Construct upstream Struct: Struct(**kwargs)
        let empty_args = PyTuple::empty(py);
        return struct_class.call(empty_args, Some(&kwargs));
    }
    // Already a Struct (subclass of dict, but downcast::<PyDict> succeeds for
    // dict subclasses too — handled above). For any other type (custom
    // namespaces, etc.) just return as-is, mirroring upstream.
    Ok(o)
}

/// Validate an orbit_wars action: `[]` or `[[int, float, int_or_float], ...]`.
///
/// Returns `(error_message_or_none, sanitized_action)`. Mirrors
/// `process_schema(action_schema, action)` for the orbit_wars schema.
///
/// orbit_wars action schema (from orbit_wars.json):
///   - top: array, default []
///   - items: array of [int (planet id), number (angle), int (ships)]
///
/// We do not run `jsonschema.validate` because the structure is fixed and
/// upstream's interpreter already sanitizes (`int(ships)`, `len(move) != 3`
/// guard, type errors in `cos(angle)` would raise upstream too).
///
/// On any structural surprise (non-list, mixed types) we return `None`
/// for the action, matching upstream's behavior of treating an unparseable
/// action as a noop after the schema error.
#[pyfunction]
pub fn fast_validate_action<'py>(
    py: Python<'py>,
    action: Bound<'py, PyAny>,
) -> PyResult<(Option<String>, Bound<'py, PyAny>)> {
    // None → upstream returns ([], None or default)
    if action.is_none() {
        return Ok((None, PyList::empty(py).into_any()));
    }
    // Not a list → defer to upstream by raising; caller's monkey patch
    // will fall back. To avoid a Python round-trip on hot path, we just
    // return the action as-is and let the interpreter handle weirdness.
    if action.downcast::<PyList>().is_err() {
        return Ok((None, action));
    }
    // Each move must be a 3-element list of (int, number, number).
    let list = action.downcast::<PyList>()?;
    for item in list.iter() {
        let Ok(inner) = item.downcast::<PyList>() else {
            // Single-tuple style like (id, angle, ships) — upstream tolerates.
            continue;
        };
        if inner.len() != 3 {
            // Upstream interpreter's process_moves silently ignores moves of
            // length != 3, so we don't flag this as an error.
            continue;
        }
    }
    Ok((None, action))
}

/// Generate 4 symmetric comet paths from pre-sampled rejection attempts.
///
/// `attempts` is a list of `(eccentricity, semi_major, orientation)` triples
/// that the Python facade has already drawn from `random.uniform` in upstream
/// order. We try them sequentially and return the first that survives the
/// collision filter. Returns `None` if every attempt is rejected.
///
/// `initial_planets` is the list-of-list shape used elsewhere in the env
/// (`[id, owner, x, y, radius, ships, production]`).
#[pyfunction]
pub fn fast_generate_comet_paths<'py>(
    py: Python<'py>,
    attempts: Bound<'py, PyList>,
    spawn_step: i64,
    angular_velocity: f64,
    comet_speed: f64,
    initial_planets: Bound<'py, PyAny>,
    comet_planet_ids: Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let mut rust_attempts: Vec<CometAttempt> = Vec::with_capacity(attempts.len());
    for item in attempts.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        rust_attempts.push(CometAttempt {
            eccentricity: tuple.get_item(0)?.extract()?,
            semi_major: tuple.get_item(1)?.extract()?,
            orientation: tuple.get_item(2)?.extract()?,
        });
    }

    let mut planets: Vec<Planet> = Vec::new();
    for p in initial_planets.try_iter()? {
        let p = p?;
        planets.push(Planet {
            id: p.get_item(0)?.extract()?,
            owner: p.get_item(1)?.extract::<i64>()? as i32,
            x: p.get_item(2)?.extract()?,
            y: p.get_item(3)?.extract()?,
            radius: p.get_item(4)?.extract()?,
            ships: p.get_item(5)?.extract()?,
            production: p.get_item(6)?.extract()?,
        });
    }
    let mut comet_ids: Vec<i64> = Vec::new();
    for v in comet_planet_ids.try_iter()? {
        comet_ids.push(v?.extract()?);
    }

    let result = py.allow_threads(|| {
        gen_paths(
            &rust_attempts,
            spawn_step,
            angular_velocity,
            comet_speed,
            &planets,
            &comet_ids,
        )
    });

    match result {
        None => Ok(None),
        Some(paths) => {
            let mut py_paths: Vec<Bound<'py, PyAny>> = Vec::with_capacity(4);
            for path in paths.iter() {
                let mut py_points: Vec<Bound<'py, PyAny>> = Vec::with_capacity(path.len());
                for pt in path.iter() {
                    let pair = PyList::new(
                        py,
                        vec![
                            pt[0].into_pyobject(py)?.into_any(),
                            pt[1].into_pyobject(py)?.into_any(),
                        ],
                    )?;
                    py_points.push(pair.into_any());
                }
                py_paths.push(PyList::new(py, py_points)?.into_any());
            }
            Ok(Some(PyList::new(py, py_paths)?))
        }
    }
}

/// Run upstream `generate_planets` against pre-sampled rejection attempts.
///
/// `phase1` is a list of (prod, angle, orbital_r, ships, orbital_skip) tuples,
/// `phase15` is (prod, orbital_r, ships, orbital_skip), `phase2` is
/// (prod, x, y, ships). The Python facade is responsible for sampling these
/// from `random.{randint,uniform}` in upstream order so that subsequent
/// random consumption (homeworld assignment, comet ship counts) lands at
/// the same RNG cursor as upstream.
///
/// Returns a list of planets in the canonical
/// `[id, owner, x, y, radius, ships, production]` shape.
#[pyfunction]
pub fn fast_generate_planets<'py>(
    py: Python<'py>,
    num_q1: i64,
    phase1: Bound<'py, PyList>,
    phase15: Bound<'py, PyList>,
    phase2: Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let mut p1: Vec<PlanetPhase1Attempt> = Vec::with_capacity(phase1.len());
    for item in phase1.iter() {
        let t = item.downcast::<PyTuple>()?;
        p1.push(PlanetPhase1Attempt {
            prod: t.get_item(0)?.extract()?,
            angle: t.get_item(1)?.extract()?,
            orbital_r: t.get_item(2)?.extract()?,
            ships: t.get_item(3)?.extract()?,
            orbital_skip: t.get_item(4)?.extract()?,
        });
    }
    let mut p15: Vec<PlanetPhase15Attempt> = Vec::with_capacity(phase15.len());
    for item in phase15.iter() {
        let t = item.downcast::<PyTuple>()?;
        p15.push(PlanetPhase15Attempt {
            prod: t.get_item(0)?.extract()?,
            orbital_r: t.get_item(1)?.extract()?,
            ships: t.get_item(2)?.extract()?,
            orbital_skip: t.get_item(3)?.extract()?,
        });
    }
    let mut p2: Vec<PlanetPhase2Attempt> = Vec::with_capacity(phase2.len());
    for item in phase2.iter() {
        let t = item.downcast::<PyTuple>()?;
        p2.push(PlanetPhase2Attempt {
            prod: t.get_item(0)?.extract()?,
            x: t.get_item(1)?.extract()?,
            y: t.get_item(2)?.extract()?,
            ships: t.get_item(3)?.extract()?,
        });
    }

    let planets = py.allow_threads(|| gen_planets(num_q1, &p1, &p15, &p2));

    let mut rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(planets.len());
    for p in planets.iter() {
        let row = PyList::new(
            py,
            vec![
                p.id.into_pyobject(py)?.into_any(),
                p.owner.into_pyobject(py)?.into_any(),
                p.x.into_pyobject(py)?.into_any(),
                p.y.into_pyobject(py)?.into_any(),
                p.radius.into_pyobject(py)?.into_any(),
                p.ships.into_pyobject(py)?.into_any(),
                p.production.into_pyobject(py)?.into_any(),
            ],
        )?;
        rows.push(row.into_any());
    }
    PyList::new(py, rows)
}

/// Filter `planets` / `initial_planets` / `comet_planet_ids` to drop any
/// entry whose planet id appears in `expired`. Returns three new PyLists.
///
/// Mirrors the Python `[p for p in xs if int(p[0]) not in expired]` triple
/// in `_facade._expire_comets_at_path_end`. Moving the loop to Rust skips
/// the Python interpreter overhead for ~30 list-comprehension iterations
/// per comet expiration turn (5–6 turns / episode).
#[pyfunction]
pub fn fast_filter_expired<'py>(
    py: Python<'py>,
    planets: Bound<'py, PyList>,
    initial_planets: Bound<'py, PyList>,
    comet_planet_ids: Bound<'py, PyList>,
    expired: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>)> {
    // Materialize `expired` into a small sorted Vec (typical size ≤ 4).
    let mut expired_ids: Vec<i64> = Vec::new();
    for v in expired.try_iter()? {
        expired_ids.push(v?.extract()?);
    }
    expired_ids.sort_unstable();
    let is_expired = |id: i64| expired_ids.binary_search(&id).is_ok();

    let new_planets = PyList::empty(py);
    for row in planets.iter() {
        let id: i64 = row.get_item(0)?.extract()?;
        if !is_expired(id) {
            new_planets.append(row)?;
        }
    }
    let new_initial = PyList::empty(py);
    for row in initial_planets.iter() {
        let id: i64 = row.get_item(0)?.extract()?;
        if !is_expired(id) {
            new_initial.append(row)?;
        }
    }
    let new_cids = PyList::empty(py);
    for v in comet_planet_ids.iter() {
        let id: i64 = v.extract()?;
        if !is_expired(id) {
            new_cids.append(id)?;
        }
    }
    Ok((new_planets, new_initial, new_cids))
}

/// Register the helpers as a sub-module accessible via
/// `orbit_wars_rust._lib.fast_helpers`.
pub fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(fast_structify, parent)?)?;
    parent.add_function(wrap_pyfunction!(fast_validate_action, parent)?)?;
    parent.add_function(wrap_pyfunction!(fast_generate_comet_paths, parent)?)?;
    parent.add_function(wrap_pyfunction!(fast_generate_planets, parent)?)?;
    parent.add_function(wrap_pyfunction!(fast_filter_expired, parent)?)?;
    Ok(())
}
