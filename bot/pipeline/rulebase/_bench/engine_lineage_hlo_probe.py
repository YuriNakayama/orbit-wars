"""Sequential-chain probe for the ENGINE-REPLAY lineage (case8) vs case1.

The allocator top-K=64 truncation fixed the geometric lineage (case1: 0.72s/turn
on GPU) but the engine-replay lineage (case4/6/8/9) still runs >13s/turn. GPU
per-turn cost is the sum of SEQUENTIAL scan steps (kernel-launch hops), so this
probe walks the jaxpr of each lineage's compute_actions and reports every
lax.scan/while with its length — exposing which chains dominate.

Run locally on CPU:  uv run python pipeline/rulebase/_bench/engine_lineage_hlo_probe.py
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
from orbit_wars_jax.reset import reset  # noqa: E402


def _scan_lengths(jaxpr) -> Counter:
    """Recursively collect lax.scan lengths (sequential step counts)."""
    found: Counter = Counter()

    def walk(jp) -> None:
        for eqn in jp.eqns:
            if eqn.primitive.name == "scan":
                found[int(eqn.params["length"])] += 1
            elif eqn.primitive.name == "while":
                found["while(?)"] += 1
            for v in eqn.params.values():
                if hasattr(v, "jaxpr"):  # ClosedJaxpr nested in params
                    walk(v.jaxpr)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if hasattr(x, "jaxpr"):
                            walk(x.jaxpr)

    walk(jaxpr.jaxpr)
    return found


def _report(name: str, fn, *args) -> None:
    jp = jax.make_jaxpr(fn)(*args)
    lengths = _scan_lengths(jp)
    total = sum(k * v for k, v in lengths.items() if isinstance(k, int))
    print(f"\n=== {name} ===")
    for k in sorted((k for k in lengths if isinstance(k, int)), reverse=True):
        print(f"  scan length {k:5d} x {lengths[k]:3d}  (seq steps {k * lengths[k]})")
    if lengths.get("while(?)"):
        print(f"  while loops (unknown len): {lengths['while(?)']}")
    print(f"  TOTAL sequential steps: {total}")


def main() -> None:
    state = reset(seed=2, num_agents=2)

    from pipeline.rulebase.case8.baseline_jax.agent_jax import (
        _modes_from_features as m8,
    )
    from pipeline.rulebase.case8.baseline_jax.agent_jax import (
        compute_actions as ca8,
    )
    from pipeline.rulebase.case8.baseline_jax.world_features import (
        build_world_features_from_state as wf8,
    )

    f8 = wf8(state, 0)
    _report("case8 (engine-replay)", ca8, f8, m8(f8))

    from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import (
        _modes_from_features as m1,
    )
    from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import (
        compute_actions as ca1,
    )
    from pipeline.rulebase.case1.baseline_jax.strict.world_features import (
        build_world_features_from_state as wf1,
    )

    f1 = wf1(state, 0)
    _report("case1 (geometric)", ca1, f1, m1(f1))


if __name__ == "__main__":
    main()
