# orbit_wars_jax

JAX-native reimplementation of the Kaggle Orbit Wars environment (`jit` + `vmap`
friendly), used by GPU rollout / featurizer paths in `bot/pipeline`.

It mirrors the dynamics of the vendored upstream Python sim
(`simulator/python/orbit_wars_vendor`) and is parity-tested against it.

## Layout

```
orbit_wars_jax/
  constants.py     Board / engine constants (CENTER, MAX_PLANETS, ...)
  state.py         EnvState (immutable, fixed-shape arrays for vmap)
  geometry.py      Swept-collision / point-to-segment helpers
  planet_gen.py    Initial planet generation
  comet_gen.py     Comet path precomputation (turns 50/150/.../450)
  combat.py        Combat resolution
  observation.py   state_to_obs (featurizer-required fields)
  reset.py         reset()
  step.py          step(), empty_actions(), MAX_LAUNCHES_PER_AGENT
```

## Import

```python
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import step, empty_actions
from orbit_wars_jax.constants import CENTER, MAX_PLANETS
```

## Test

```bash
# from repo root
dev/test-bot         # includes this package's parity tests (run under bot/)
```

Parity tests against the upstream sim live in `bot/tests/unit/jax_env/`
(they import both `orbit_wars_jax` and `orbit_wars_vendor`).
