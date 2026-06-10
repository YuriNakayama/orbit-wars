"""Attribute case8's sequential scans to their source component.

case8 compute_actions has 6,831 sequential scan steps (110x44 + 219x8 + misc) vs
case1's 1,978 — this probe runs the jaxpr scan-length census per COMPONENT
(capture grid / snipe grid / harass grid / allocator) so the reduction work
targets the right chains.

Run:  uv run python pipeline/rulebase/_bench/engine_scan_attribution.py
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
from orbit_wars_jax.reset import reset  # noqa: E402


def _scan_lengths(jp) -> Counter:
    found: Counter = Counter()

    def walk(j) -> None:
        for eqn in j.eqns:
            if eqn.primitive.name == "scan":
                found[int(eqn.params["length"])] += 1
            for v in eqn.params.values():
                if hasattr(v, "jaxpr"):
                    walk(v.jaxpr)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if hasattr(x, "jaxpr"):
                            walk(x.jaxpr)

    walk(jp.jaxpr)
    return found


def _census(name: str, fn, *args) -> None:
    c = _scan_lengths(jax.make_jaxpr(fn)(*args))
    total = sum(k * v for k, v in c.items())
    parts = ", ".join(f"{k}x{v}" for k, v in sorted(c.items(), reverse=True))
    print(f"{name:28s} total={total:6d}  [{parts}]")


def main() -> None:
    state = reset(seed=2, num_agents=2)

    import pipeline.rulebase.case8.baseline_jax.missions_capture_jax as mc8
    from pipeline.rulebase.case8.baseline_jax.agent_jax import (
        _combine_single_table,
        _modes_from_features,
    )
    from pipeline.rulebase.case8.baseline_jax.allocator_jax import (
        _empty_carry,
        _run_mission_scan,
        run_followup_pass,
    )
    from pipeline.rulebase.case8.baseline_jax.world_features import (
        build_world_features_from_state,
    )

    f = build_world_features_from_state(state, 0)
    m = _modes_from_features(f)
    tl = mc8._base_timelines(f)

    _census("_base_timelines", lambda feats: mc8._base_timelines(feats), f)
    _census("build_capture_grid", mc8.build_capture_grid, f, m)
    _census("build_snipe_grid", mc8.build_snipe_grid, f, m)
    _census("build_harass_grid", mc8.build_harass_grid, f, m)

    cap = mc8.build_capture_grid(f, m)
    sn = mc8.build_snipe_grid(f, m)
    ha = mc8.build_harass_grid(f, m)
    table = _combine_single_table(cap, sn, ha)
    _census(
        "_run_mission_scan",
        lambda t, feats, modes: _run_mission_scan(t, _empty_carry(), feats, modes),
        table,
        f,
        m,
    )
    carry = _run_mission_scan(table, _empty_carry(), f, m)
    _census(
        "run_followup_pass",
        lambda c, feats, modes: run_followup_pass(c, feats, modes),
        carry,
        f,
        m,
    )

    # Drill into one capture cell: where do its scans come from?
    import jax.numpy as jnp

    _census(
        "  one _capture_cell",
        lambda feats, modes: mc8._capture_cell(
            feats, modes, tl[0], tl[1], jnp.int32(0), jnp.int32(1)
        ),
        f,
        m,
    )
    from pipeline.rulebase.case8.baseline_jax.missions_capture_jax import (
        _plan_shot_cell,
    )

    _census(
        "  one _plan_shot_cell",
        lambda feats: _plan_shot_cell(feats, jnp.int32(0), jnp.int32(1), jnp.int32(5)),
        f,
    )


if __name__ == "__main__":
    main()
