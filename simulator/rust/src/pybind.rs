//! PyO3 ⇄ `OrbitWarsState` conversion helpers.
//!
//! The Kaggle environment passes the interpreter:
//!   - `state`: `[ {observation: {...}, action: [...], status, reward}, ... ]`
//!   - `env`: an object exposing `configuration` (episodeSteps, shipSpeed, cometSpeed).
//!
//! We read enough out of `state[0].observation` to populate `OrbitWarsState`,
//! run `interpreter::step`, and write the resulting world back into the same
//! Python objects (mutating planets/fleets/etc. in place so the framework's
//! `env.steps[-1]` reflects the new state).
//!
//! The upstream Python code uses `getattr(d, key, default)` style access
//! because the observation can be either a `dict` or a `SimpleNamespace`. To
//! match that, we accept both via `try_get_attr_or_item` below.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::state::{CometGroup, Fleet, Move, OrbitWarsState, Planet};

/// Read `obj.<key>` falling back to `obj[<key>]`. Returns `None` if neither
/// path resolves.
fn get_attr_or_item<'py>(
    obj: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(value) = obj.getattr(key) {
        if !value.is_none() {
            return Ok(Some(value));
        }
    }
    if let Ok(value) = obj.get_item(key) {
        if !value.is_none() {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Set `obj.<key> = value` falling back to `obj[<key>] = value`.
fn set_attr_or_item<'py>(
    obj: &Bound<'py, PyAny>,
    key: &str,
    value: Bound<'py, PyAny>,
) -> PyResult<()> {
    if obj.hasattr(key)? {
        obj.setattr(key, value)?;
        return Ok(());
    }
    obj.set_item(key, value)?;
    Ok(())
}

/// Determine once whether an object supports attribute assignment (`Struct`,
/// SimpleNamespace, etc.) or only item assignment (plain dict). Returns
/// `true` if `setattr(obj, key, ...)` is the right path, `false` for
/// `obj[key] = ...`. We probe a sentinel attribute to avoid the cost of
/// `hasattr(specific_key)` once per write.
fn supports_setattr<'py>(obj: &Bound<'py, PyAny>) -> bool {
    // PyO3's `getattr` triggers `__getattr__` for dict subclasses (like
    // upstream's `Struct`) when the attribute exists in `__dict__`. We use
    // `__class__` (always present) only to confirm we have a Python object;
    // the dict / Struct branch is decided by `hasattr` for a benign field
    // we know to be in the observation.
    obj.hasattr("step").unwrap_or(false)
}

/// Pre-resolved per-key setter — picks attr-or-item once, then applies it
/// without rechecking on subsequent calls.
#[inline]
fn set_field_with_mode<'py>(
    obj: &Bound<'py, PyAny>,
    use_attr: bool,
    key: &str,
    value: Bound<'py, PyAny>,
) -> PyResult<()> {
    if use_attr {
        // Struct / SimpleNamespace: setattr keeps both attr and item slots
        // in sync (Struct's __setattr__ writes to dict storage too).
        obj.setattr(key, value)?;
    } else {
        obj.set_item(key, value)?;
    }
    Ok(())
}

#[inline]
fn extract_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value.extract::<f64>()
}

#[inline]
fn extract_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    // Most planet/fleet integer fields arrive as Python `int` (PyLong),
    // so `extract::<i64>` succeeds on the first try. Only fall back when
    // the value is a `float` (rare — comet ship counts etc. all int).
    if let Ok(n) = value.extract::<i64>() {
        return Ok(n);
    }
    let f: f64 = value.extract()?;
    Ok(f as i64)
}

/// Collect a 7-element row (planet/fleet) into a fixed-size array using
/// `try_iter` so the iterator object is constructed once. This is faster
/// than 7 successive `get_item(i)` calls for `PyList` because each
/// `get_item` does a bounds-check + reference clone, while iteration
/// hands out borrowed views.
fn collect_row7<'py>(item: &Bound<'py, PyAny>) -> PyResult<[Bound<'py, PyAny>; 7]> {
    // Try the typed-PyList fast path first; tuples and other sequences
    // fall back to the iterator path below.
    if let Ok(list) = item.downcast::<PyList>() {
        if list.len() == 7 {
            return Ok([
                list.get_item(0)?,
                list.get_item(1)?,
                list.get_item(2)?,
                list.get_item(3)?,
                list.get_item(4)?,
                list.get_item(5)?,
                list.get_item(6)?,
            ]);
        }
    }
    // Generic path via try_iter — works for tuple, namedtuple, custom seq.
    let mut iter = item.try_iter()?;
    let mut out: [Option<Bound<'py, PyAny>>; 7] = Default::default();
    for slot in out.iter_mut() {
        *slot = Some(
            iter.next()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("row too short"))??,
        );
    }
    let arr: [Bound<'py, PyAny>; 7] = out.map(|opt| opt.expect("loop above filled all slots"));
    Ok(arr)
}

/// Read a list-of-rows into `out` planets vec. Uses a typed-PyList fast
/// path with direct indexing (skips iterator setup for the common case
/// where upstream observations always hand us PyList).
fn read_into_planets(obj: &Bound<'_, PyAny>, out: &mut Vec<Planet>) -> PyResult<()> {
    if let Ok(list) = obj.downcast::<PyList>() {
        let n = list.len();
        out.reserve(n);
        for i in 0..n {
            let item = list.get_item(i)?;
            out.push(read_planet_any(&item)?);
        }
        return Ok(());
    }
    // Generic path: try_iter handles tuples / namedtuples / custom seq.
    if let Ok(n) = obj.len() {
        out.reserve(n);
    }
    for item in obj.try_iter()? {
        out.push(read_planet_any(&item?)?);
    }
    Ok(())
}

fn read_into_fleets(obj: &Bound<'_, PyAny>, out: &mut Vec<Fleet>) -> PyResult<()> {
    if let Ok(list) = obj.downcast::<PyList>() {
        let n = list.len();
        out.reserve(n);
        for i in 0..n {
            let item = list.get_item(i)?;
            out.push(read_fleet_any(&item)?);
        }
        return Ok(());
    }
    if let Ok(n) = obj.len() {
        out.reserve(n);
    }
    for item in obj.try_iter()? {
        out.push(read_fleet_any(&item?)?);
    }
    Ok(())
}

fn read_planet_any(item: &Bound<'_, PyAny>) -> PyResult<Planet> {
    let row = collect_row7(item)?;
    Ok(Planet {
        id: extract_i64(&row[0])?,
        owner: extract_i64(&row[1])? as i32,
        x: extract_f64(&row[2])?,
        y: extract_f64(&row[3])?,
        radius: extract_f64(&row[4])?,
        ships: extract_i64(&row[5])?,
        production: extract_i64(&row[6])?,
    })
}

fn read_fleet_any(item: &Bound<'_, PyAny>) -> PyResult<Fleet> {
    let row = collect_row7(item)?;
    Ok(Fleet {
        id: extract_i64(&row[0])?,
        owner: extract_i64(&row[1])? as i32,
        x: extract_f64(&row[2])?,
        y: extract_f64(&row[3])?,
        angle: extract_f64(&row[4])?,
        from_planet_id: extract_i64(&row[5])?,
        ships: extract_i64(&row[6])?,
    })
}

fn read_comets<'py>(value: &Bound<'py, PyAny>) -> PyResult<(Vec<CometGroup>, Vec<i64>)> {
    let mut comets: Vec<CometGroup> = Vec::new();
    let mut comet_ids: Vec<i64> = Vec::new();
    // Fast path: empty list / no comets (~70% of turns).
    if let Ok(n) = value.len() {
        if n == 0 {
            return Ok((comets, comet_ids));
        }
        comets.reserve(n);
    }
    let groups = value.try_iter()?;
    for group in groups {
        let group = group?;
        let planet_ids_obj = get_attr_or_item(&group, "planet_ids")?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("comet group missing planet_ids")
        })?;
        let mut planet_ids: Vec<i64> = Vec::new();
        for pid in planet_ids_obj.try_iter()? {
            planet_ids.push(extract_i64(&pid?)?);
        }
        comet_ids.extend(planet_ids.iter().copied());

        let paths_obj = get_attr_or_item(&group, "paths")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("comet group missing paths"))?;
        let mut paths: Vec<Vec<[f64; 2]>> = Vec::new();
        for path in paths_obj.try_iter()? {
            let path = path?;
            let mut points: Vec<[f64; 2]> = Vec::new();
            for point in path.try_iter()? {
                let point = point?;
                let x = extract_f64(&point.get_item(0)?)?;
                let y = extract_f64(&point.get_item(1)?)?;
                points.push([x, y]);
            }
            paths.push(points);
        }

        let path_index = match get_attr_or_item(&group, "path_index")? {
            Some(v) => extract_i64(&v)?,
            None => -1,
        };

        comets.push(CometGroup {
            planet_ids,
            paths,
            path_index,
        });
    }
    Ok((comets, comet_ids))
}

fn read_actions(state_list: &Bound<'_, PyList>, num_agents: usize) -> PyResult<Vec<Vec<Move>>> {
    let mut actions: Vec<Vec<Move>> = Vec::with_capacity(num_agents);
    for i in 0..num_agents {
        let agent_state = state_list.get_item(i)?;
        let action_obj = match get_attr_or_item(&agent_state, "action")? {
            Some(v) => v,
            None => {
                actions.push(Vec::new());
                continue;
            }
        };
        // Pre-size moves vec when we can read the length cheaply (most actions
        // are PyList). Falls back to default growth otherwise.
        let cap = action_obj.len().unwrap_or(0);
        let mut moves: Vec<Move> = Vec::with_capacity(cap);
        for mv in action_obj.try_iter()? {
            let mv = mv?;
            // Each move is a 3-tuple [from, angle, ships]. Use a single
            // try_iter pull to grab all 3 elements with minimal cross-language
            // calls. Skip silently when the move shape is wrong (matches
            // upstream `if len(move) != 3: continue`).
            if let Ok((from, angle, ships)) = read_move_triple(&mv) {
                moves.push(Move {
                    from_planet_id: from,
                    angle,
                    ships,
                });
            }
        }
        actions.push(moves);
    }
    Ok(actions)
}

/// Pull three scalars out of a move tuple/list. Returns Err if shape
/// mismatches (caller treats as a noop). Uses `try_iter` once instead of
/// three `get_item` calls.
fn read_move_triple(mv: &Bound<'_, PyAny>) -> PyResult<(i64, f64, i64)> {
    // Fast path: PyList of 3 — direct indexed reads beat iterator setup.
    if let Ok(list) = mv.downcast::<PyList>() {
        if list.len() == 3 {
            let from = extract_i64(&list.get_item(0)?)?;
            let angle = extract_f64(&list.get_item(1)?)?;
            let ships = extract_i64(&list.get_item(2)?)?;
            return Ok((from, angle, ships));
        }
        return Err(pyo3::exceptions::PyValueError::new_err(
            "move not 3 elements",
        ));
    }
    // Generic path: tuple, namedtuple, etc.
    let mut iter = mv.try_iter()?;
    let from = extract_i64(
        &iter
            .next()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("move empty"))??,
    )?;
    let angle = extract_f64(
        &iter
            .next()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("move missing angle"))??,
    )?;
    let ships = extract_i64(
        &iter
            .next()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("move missing ships"))??,
    )?;
    if iter.next().is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "move has extra elements",
        ));
    }
    Ok((from, angle, ships))
}

/// Hydrate `OrbitWarsState` from the Python `state` list and `env`.
pub fn pylist_to_state<'py>(
    state_list: &Bound<'py, PyList>,
    env: &Bound<'py, PyAny>,
) -> PyResult<(OrbitWarsState, Vec<Vec<Move>>, Bound<'py, PyAny>)> {
    let num_agents = state_list.len();
    let obs0 = state_list.get_item(0)?;
    let observation = get_attr_or_item(&obs0, "observation")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("state[0] missing 'observation'"))?;

    let configuration = get_attr_or_item(env, "configuration")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("env missing 'configuration'"))?;
    let episode_steps = get_attr_or_item(&configuration, "episodeSteps")?
        .map(|v| extract_i64(&v))
        .transpose()?
        .unwrap_or(500);
    let ship_speed = get_attr_or_item(&configuration, "shipSpeed")?
        .map(|v| extract_f64(&v))
        .transpose()?
        .unwrap_or(6.0);
    let comet_speed = get_attr_or_item(&configuration, "cometSpeed")?
        .map(|v| extract_f64(&v))
        .transpose()?
        .unwrap_or(4.0);

    let mut state = OrbitWarsState::new(num_agents, 0);
    state.episode_steps = episode_steps;
    state.ship_speed = ship_speed;
    state.comet_speed = comet_speed;

    state.angular_velocity = get_attr_or_item(&observation, "angular_velocity")?
        .map(|v| extract_f64(&v))
        .transpose()?
        .unwrap_or(0.0);
    state.step = get_attr_or_item(&observation, "step")?
        .map(|v| extract_i64(&v))
        .transpose()?
        .unwrap_or(0);
    state.next_fleet_id = get_attr_or_item(&observation, "next_fleet_id")?
        .map(|v| extract_i64(&v))
        .transpose()?
        .unwrap_or(0);

    // Read planets / initial_planets / fleets. We probe for the typed
    // PyList fast path first (avoids the `try_iter` machinery setup) and
    // fall back to the generic iterator for tuples / namedtuples.
    if let Some(planets_obj) = get_attr_or_item(&observation, "planets")? {
        read_into_planets(&planets_obj, &mut state.planets)?;
    }
    if let Some(initial_obj) = get_attr_or_item(&observation, "initial_planets")? {
        read_into_planets(&initial_obj, &mut state.initial_planets)?;
    } else {
        state.initial_planets = state.planets.clone();
    }
    if let Some(fleets_obj) = get_attr_or_item(&observation, "fleets")? {
        read_into_fleets(&fleets_obj, &mut state.fleets)?;
    }
    // Comets fast path: skip both `comets` and `comet_planet_ids` reads
    // entirely when the comets list is empty (the typical case for
    // most turns — comets only exist between spawn (50/150/250/350/450)
    // and path expiration ~30 turns later).
    let comets_obj = get_attr_or_item(&observation, "comets")?;
    let comets_empty = match &comets_obj {
        Some(obj) => obj.len().map(|n| n == 0).unwrap_or(false),
        None => true,
    };
    if !comets_empty {
        if let Some(obj) = comets_obj {
            let (comets, ids) = read_comets(&obj)?;
            state.comets = comets;
            state.comet_planet_ids = ids;
        }
        // Override comet_planet_ids when explicitly provided and non-empty.
        if let Some(ids_obj) = get_attr_or_item(&observation, "comet_planet_ids")? {
            let len_hint = ids_obj.len().unwrap_or(0);
            if len_hint > 0 {
                let mut ids: Vec<i64> = Vec::with_capacity(len_hint);
                for v in ids_obj.try_iter()? {
                    ids.push(extract_i64(&v?)?);
                }
                if !ids.is_empty() {
                    state.comet_planet_ids = ids;
                }
            }
        }
    }

    let actions = read_actions(state_list, num_agents)?;
    Ok((state, actions, observation))
}

/// Build the canonical 7-field planet list. We pass a fixed-size array to
/// `PyList::new` instead of building a heap-allocated `Vec` first — saves
/// one allocation per planet × planet_count × turn (= ~32 × 500 = 16000
/// per episode).
#[inline]
fn planet_to_pylist<'py>(py: Python<'py>, p: &Planet) -> PyResult<Bound<'py, PyList>> {
    let items: [Bound<'py, PyAny>; 7] = [
        p.id.into_pyobject(py)?.into_any(),
        p.owner.into_pyobject(py)?.into_any(),
        p.x.into_pyobject(py)?.into_any(),
        p.y.into_pyobject(py)?.into_any(),
        p.radius.into_pyobject(py)?.into_any(),
        p.ships.into_pyobject(py)?.into_any(),
        p.production.into_pyobject(py)?.into_any(),
    ];
    PyList::new(py, items)
}

#[inline]
fn fleet_to_pylist<'py>(py: Python<'py>, f: &Fleet) -> PyResult<Bound<'py, PyList>> {
    let items: [Bound<'py, PyAny>; 7] = [
        f.id.into_pyobject(py)?.into_any(),
        f.owner.into_pyobject(py)?.into_any(),
        f.x.into_pyobject(py)?.into_any(),
        f.y.into_pyobject(py)?.into_any(),
        f.angle.into_pyobject(py)?.into_any(),
        f.from_planet_id.into_pyobject(py)?.into_any(),
        f.ships.into_pyobject(py)?.into_any(),
    ];
    PyList::new(py, items)
}

fn comets_to_pyobj<'py>(py: Python<'py>, comets: &[CometGroup]) -> PyResult<Bound<'py, PyList>> {
    // Fast path: empty comets list (~70% of turns).
    if comets.is_empty() {
        return PyList::new(py, std::iter::empty::<i64>());
    }
    let mut groups: Vec<Bound<'py, PyAny>> = Vec::with_capacity(comets.len());
    for g in comets.iter() {
        let group_dict = PyDict::new(py);
        let pids = PyList::new(py, g.planet_ids.iter().copied())?;
        group_dict.set_item("planet_ids", pids)?;

        let mut paths: Vec<Bound<'py, PyAny>> = Vec::with_capacity(g.paths.len());
        for path in g.paths.iter() {
            let mut pts: Vec<Bound<'py, PyAny>> = Vec::with_capacity(path.len());
            for pt in path.iter() {
                let pair: [Bound<'py, PyAny>; 2] = [
                    pt[0].into_pyobject(py)?.into_any(),
                    pt[1].into_pyobject(py)?.into_any(),
                ];
                let pair_list = PyList::new(py, pair)?;
                pts.push(pair_list.into_any());
            }
            paths.push(PyList::new(py, pts)?.into_any());
        }
        group_dict.set_item("paths", PyList::new(py, paths)?)?;
        group_dict.set_item("path_index", g.path_index)?;
        groups.push(group_dict.into_any());
    }
    PyList::new(py, groups)
}

/// Mutate the `state` list in place so it reflects `world` after `step`.
pub fn write_state_back<'py>(
    py: Python<'py>,
    state_list: &Bound<'py, PyList>,
    observation: &Bound<'py, PyAny>,
    world: &OrbitWarsState,
) -> PyResult<()> {
    // 1. Build PyList versions of planets / fleets / comets / etc.
    let mut planet_rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(world.planets.len());
    for p in world.planets.iter() {
        planet_rows.push(planet_to_pylist(py, p)?.into_any());
    }
    let planets = PyList::new(py, planet_rows)?;

    let mut initial_rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(world.initial_planets.len());
    for p in world.initial_planets.iter() {
        initial_rows.push(planet_to_pylist(py, p)?.into_any());
    }
    let initial_planets = PyList::new(py, initial_rows)?;

    let mut fleet_rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(world.fleets.len());
    for f in world.fleets.iter() {
        fleet_rows.push(fleet_to_pylist(py, f)?.into_any());
    }
    let fleets = PyList::new(py, fleet_rows)?;

    let comets = comets_to_pyobj(py, &world.comets)?;
    let comet_ids = PyList::new(py, world.comet_planet_ids.iter().copied())?;

    let next_fleet_id_py = world.next_fleet_id.into_pyobject(py)?.into_any();
    let angular_velocity_py = world.angular_velocity.into_pyobject(py)?.into_any();
    let planets_any = planets.into_any();
    let initial_any = initial_planets.into_any();
    let fleets_any = fleets.into_any();
    let comets_any = comets.into_any();
    let comet_ids_any = comet_ids.into_any();

    // Decide attr-vs-item ONCE for observation 0; reuse for every field.
    let obs0_use_attr = supports_setattr(observation);

    // 2. Set them on observation 0 + mirror to other agents in one pass.
    // We always need a clone for each set (PyAny references are reference
    // counted; setattr/setitem moves the binding). To avoid the final
    // unnecessary clone on the LAST agent, we move the original and clone
    // only for prior agents.
    let agents = world.num_agents;
    if agents == 1 {
        // Single-agent: move references straight into observation, no clones.
        set_field_with_mode(observation, obs0_use_attr, "planets", planets_any)?;
        set_field_with_mode(observation, obs0_use_attr, "initial_planets", initial_any)?;
        set_field_with_mode(observation, obs0_use_attr, "fleets", fleets_any)?;
        set_field_with_mode(
            observation,
            obs0_use_attr,
            "next_fleet_id",
            next_fleet_id_py,
        )?;
        set_field_with_mode(
            observation,
            obs0_use_attr,
            "angular_velocity",
            angular_velocity_py,
        )?;
        set_field_with_mode(observation, obs0_use_attr, "comets", comets_any)?;
        set_field_with_mode(
            observation,
            obs0_use_attr,
            "comet_planet_ids",
            comet_ids_any,
        )?;
    } else {
        // Multi-agent: gather every observation up-front so we can move the
        // last reference instead of cloning.
        let mut obs_handles: Vec<(Bound<'py, PyAny>, bool)> = Vec::with_capacity(agents);
        obs_handles.push((observation.clone(), obs0_use_attr));
        for i in 1..agents {
            let agent_state = state_list.get_item(i)?;
            let other_obs = match get_attr_or_item(&agent_state, "observation")? {
                Some(v) => v,
                None => continue,
            };
            let use_attr = supports_setattr(&other_obs);
            obs_handles.push((other_obs, use_attr));
        }

        // Each field gets one Bound::clone per observation. The clone cost
        // (a refcount increment) is unavoidable since `set_field_with_mode`
        // takes ownership; the previous code did the same number of clones.
        // Hoisting the iteration order saves a per-agent get_item on
        // `observation`: agent 0 is already in `obs_handles[0]`.
        for (obs_ref, use_attr) in obs_handles.iter() {
            set_field_with_mode(obs_ref, *use_attr, "planets", planets_any.clone())?;
            set_field_with_mode(obs_ref, *use_attr, "initial_planets", initial_any.clone())?;
            set_field_with_mode(obs_ref, *use_attr, "fleets", fleets_any.clone())?;
            set_field_with_mode(
                obs_ref,
                *use_attr,
                "next_fleet_id",
                next_fleet_id_py.clone(),
            )?;
            set_field_with_mode(
                obs_ref,
                *use_attr,
                "angular_velocity",
                angular_velocity_py.clone(),
            )?;
            set_field_with_mode(obs_ref, *use_attr, "comets", comets_any.clone())?;
            set_field_with_mode(
                obs_ref,
                *use_attr,
                "comet_planet_ids",
                comet_ids_any.clone(),
            )?;
        }
    }

    // 4. Population termination state on per-agent status / reward.
    if world.done {
        for i in 0..world.num_agents {
            let agent_state = state_list.get_item(i)?;
            let status_str = pyo3::types::PyString::new(py, "DONE");
            set_attr_or_item(&agent_state, "status", status_str.into_any())?;
            let reward = world.rewards.get(i).copied().unwrap_or(0);
            set_attr_or_item(
                &agent_state,
                "reward",
                (reward as i64).into_pyobject(py)?.into_any(),
            )?;
        }
    }

    Ok(())
}
