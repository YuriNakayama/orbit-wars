# `simulator/python/` — Vendored Orbit Wars (Apache-2.0)

This directory holds an **unmodified copy** of
[`kaggle_environments/envs/orbit_wars`](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars)
shipped with `kaggle-environments==1.28.0` (May 2026).

## Why vendor?

Two reasons:

1. **Reference implementation** for the Rust port in `simulator/rust/`. The
   Python interpreter is the source-of-truth used by parity tests
   (`simulator/rust/python/tests/test_parity.py`).
2. **Runtime fallback**. The facade in `orbit_wars_rust` dispatches to this
   vendored copy when the env var `ORBIT_WARS_BACKEND=python` is set, which is
   how we keep the Rust simulator opt-out-able.

## Contents

```
simulator/python/
├── LICENSE                           # Apache-2.0 from kaggle-environments
├── NOTICE                            # Provenance: source URL + version
├── README.md                         # this file
├── pyproject.toml                    # hatch build (orbit-wars-vendor)
└── orbit_wars_vendor/
    ├── __init__.py                   # re-exports interpreter / Planet / Fleet / ...
    ├── orbit_wars.py                 # 806 lines (verbatim copy)
    ├── orbit_wars.json
    ├── orbit_wars.js
    ├── README.md                     # original kaggle-environments README
    └── tests/
        ├── __init__.py
        └── test_orbit_wars.py        # 585 lines, 23 cases (verbatim copy)
```

## Running the vendored tests

From the repo root:

```bash
(cd simulator/python && uv run pytest orbit_wars_vendor/tests -q)
```

The 23 cases must be green before any Rust port work proceeds.

## Upgrading the vendored copy

When `kaggle-environments` is bumped in `bot/pyproject.toml`, follow the
checklist in `NOTICE`. **Do not modify** the four vendored files in place —
the only valid edit pattern is "delete + re-copy from upstream".
