//! Rust port of the Kaggle Orbit Wars interpreter, exposed to Python via PyO3.
//!
//! This crate is the implementation backing `orbit_wars_rust._lib` (see
//! `pyproject.toml`). The Python facade in `python/orbit_wars_rust/` is what
//! actually registers the env name with `kaggle_environments`; this crate
//! only provides the per-step interpreter callable.

#![forbid(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::PyList;

pub mod combat;
pub mod fast_helpers;
pub mod generation;
pub mod interpreter;
pub mod physics;
pub mod pybind;
pub mod rng;
pub mod state;

/// Interpreter entry point called by `kaggle_environments.Environment.step`.
///
/// Reads `state` / `env`, runs `interpreter::step`, and writes the new world
/// back into the same Python objects. Returns the same `state` list (the
/// Kaggle framework re-uses the reference).
#[pyfunction]
#[pyo3(name = "interpreter")]
fn py_interpreter<'py>(
    py: Python<'py>,
    state: Bound<'py, PyList>,
    env: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    // The upstream interpreter early-returns when env.done is set. Mirror it.
    if let Ok(done_obj) = env.getattr("done") {
        if let Ok(done) = done_obj.extract::<bool>() {
            if done {
                return Ok(state);
            }
        }
    }
    let (mut world, actions, observation) = pybind::pylist_to_state(&state, &env)?;
    py.allow_threads(|| interpreter::step(&mut world, &actions));
    pybind::write_state_back(py, &state, &observation, &world)?;
    Ok(state)
}

#[pymodule]
fn _lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_interpreter, m)?)?;
    fast_helpers::register(py, m)?;
    Ok(())
}
