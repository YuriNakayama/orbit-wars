# Rust Simulator — Architecture Design

## Top-Level Directory Layout (after rename + simulator/)

```
orbit-wars/
├── bot/                              # 旧 backend/ (rename only)
│   ├── pyproject.toml                # 既存。orbit-wars-rust への local 依存を追加
│   ├── uv.lock
│   ├── src/                          # 既存 (agents / dataset / submit / vast / ...)
│   ├── pipeline/                     # 既存 (rulebase / imitation / ...)
│   └── tests/                        # 既存
├── simulator/                        # ← 新規
│   ├── README.md                     # 全体概要 (Python/Rust の分担を説明)
│   ├── python/                       # 公式 vendored copy
│   │   ├── pyproject.toml            # hatch (orbit-wars-vendor)
│   │   ├── LICENSE                   # Apache-2.0 (Kaggle/kaggle-environments より)
│   │   ├── NOTICE                    # 出典 URL + commit SHA
│   │   ├── README.md                 # vendor copy 注記
│   │   └── orbit_wars_vendor/
│   │       ├── __init__.py           # re-export interpreter / generate_planets / ...
│   │       ├── orbit_wars.py         # 公式 806行 (無修正)
│   │       ├── orbit_wars.json
│   │       ├── orbit_wars.js
│   │       └── tests/
│   │           └── test_orbit_wars.py  # 公式 585行 (pytest 互換に minor 修正)
│   └── rust/                         # PyO3 + maturin (hybrid layout)
│       ├── Cargo.toml
│       ├── Cargo.lock
│       ├── pyproject.toml            # [build-system] requires = ["maturin"]
│       ├── README.md
│       ├── src/
│       │   ├── lib.rs                # #[pymodule] 定義 + Python 側 hook
│       │   ├── state.rs              # OrbitWarsState 構造体
│       │   ├── physics.rs            # fleet / planet motion + collision
│       │   ├── combat.rs             # combat resolution
│       │   ├── generation.rs         # generate_planets / generate_comet_paths
│       │   ├── interpreter.rs        # interpreter(state, env) -> state
│       │   ├── rng.rs                # ChaCha12 wrapper
│       │   └── pybind.rs             # PyO3 conversion helpers
│       ├── python/
│       │   └── orbit_wars_rust/
│       │       ├── __init__.py       # import 時に kaggle_environments.register()
│       │       ├── _facade.py        # ENV var による rust/python switching
│       │       └── _vendor_bridge.py # python backend 選択時に simulator/python を呼ぶ
│       ├── tests/                    # cargo test (Rust unit + integration)
│       │   ├── test_combat.rs
│       │   ├── test_motion.rs
│       │   ├── test_generation.rs
│       │   └── test_full_episode.rs
│       └── benches/
│           └── parity.rs             # criterion benchmark
├── dev/
│   ├── setup                         # bot rename + maturin develop --release を追加
│   ├── test-bot                      # 旧 test-backend
│   ├── lint                          # cargo fmt / clippy も実行
│   ├── format                        # cargo fmt も実行
│   ├── dvc / vast / submit / create-worktree / runtime-build / sync-data
├── data/                             # symlink (既存)
├── docs/
│   └── plans/rust-simulator/         # 本プラン
├── infra/                            # Dockerfile の WORKDIR / COPY を bot に
├── .github/workflows/
│   ├── ci-bot.yml                    # 旧 ci-backend.yml
│   ├── build-push.yml                # paths: bot/**, working-directory: bot
│   └── cron-kaggle-scrape.yml        # working-directory: bot
├── .claude/
│   ├── CLAUDE.md                     # backend → bot, simulator/ を追加
│   └── rules/bot/{pipeline,tests}.md # 旧 rules/backend/
├── dvc.yaml                          # cmd: uv run --directory bot ..., deps: bot/...
├── params.yaml
└── README.md
```

---

## Component Architecture

### Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ bot/ (旧 backend/)                                                │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ src/{dataset/selfplay, evaluation, agents, submit, ...}  │    │
│  │   import kaggle_environments                              │    │
│  │   import orbit_wars_rust   ← side-effect で register()    │    │
│  │   make("orbit_wars", config) → env.run / env.step         │    │
│  └─────────────────────┬────────────────────────────────────┘    │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────┐
            │ kaggle_environments (公式)    │
            │   register(name, env_dict)   │
            │   make() → Environment       │
            │   Environment.run/step       │
            │   ※ interpreter は差替え対象 │
            └──────────────┬───────────────┘
                           │ interpreter callback
                           ▼
            ┌──────────────────────────────────────┐
            │ orbit_wars_rust (Python module)      │
            │   _facade.interpreter(state, env)    │
            │     ├─ ORBIT_WARS_BACKEND=rust →     │
            │     │     PyO3 → Rust _lib           │
            │     └─ ORBIT_WARS_BACKEND=python →   │
            │           simulator/python/...       │
            └─────┬────────────────────┬───────────┘
                  │                    │
                  ▼                    ▼
   ┌──────────────────────┐    ┌──────────────────────────┐
   │ simulator/rust/      │    │ simulator/python/        │
   │ (Rust core)          │    │ (Apache-2.0 vendored)    │
   │  - state.rs          │    │  orbit_wars_vendor/      │
   │  - physics.rs        │    │   orbit_wars.py (806行)  │
   │  - combat.rs         │    │   orbit_wars.json        │
   │  - generation.rs     │    │   tests/test_orbit_wars  │
   │  - rng.rs (ChaCha12) │    └──────────────────────────┘
   │  - interpreter.rs    │
   │  - pybind.rs         │
   └──────────────────────┘
```

### Backend selection flow (runtime)

1. `import orbit_wars_rust` → `__init__.py` の side-effect で `kaggle_environments.register("orbit_wars", {...})` が呼ばれる。
2. 既存コードが `make("orbit_wars", ...)` すると `Environment.interpreter` に `_facade.interpreter` が入る。
3. `Environment.step()` から `_facade.interpreter(state, env)` が呼ばれ、`os.environ.get("ORBIT_WARS_BACKEND", "rust")` で Rust か Python かを分岐。
4. Rust 側は **`OrbitWarsState` を Rust 内に保持**。Python から渡された `state[0].observation.planets` を見て、(a) state が空なら新規生成して PyDict 化して `state` に書き戻し、(b) state が埋まっていれば既存 Rust state を更新して PyDict に反映する。
5. `Environment.toJSON()` は公式の挙動どおり `env.steps` を JSON 化。Rust 側で書き戻した PyDict がそのまま使われるため互換維持。

---

## Backend (Rust) Design

### `simulator/rust/Cargo.toml`

```toml
[package]
name = "orbit_wars_rust"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[lib]
name = "orbit_wars_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
rand = "0.8"
rand_chacha = "0.3"
thiserror = "1"

[dev-dependencies]
criterion = "0.5"
proptest = "1"

[[bench]]
name = "parity"
harness = false

[profile.release]
lto = "fat"
codegen-units = 1
```

### `simulator/rust/pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "orbit-wars-rust"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
description = "Rust implementation of Kaggle Orbit Wars simulator"
license = {text = "Apache-2.0"}
dependencies = ["kaggle-environments>=1.17.0"]

[tool.maturin]
python-source = "python"
module-name = "orbit_wars_rust._lib"
features = ["pyo3/extension-module"]
```

### Key Rust types (`src/state.rs`)

```rust
#[derive(Clone, Debug)]
pub struct Planet {
    pub id: i64,
    pub owner: i32,
    pub x: f64, pub y: f64,
    pub radius: f64,
    pub ships: i64,
    pub production: i64,
}

#[derive(Clone, Debug)]
pub struct Fleet {
    pub id: i64, pub owner: i32,
    pub x: f64, pub y: f64, pub angle: f64,
    pub from_planet_id: i64,
    pub ships: i64,
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub planet_ids: Vec<i64>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i64,
}

#[derive(Clone, Debug)]
pub struct OrbitWarsState {
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: i64,
    pub comets: Vec<CometGroup>,
    pub comet_planet_ids: Vec<i64>,
    pub step: i64,
    pub rng: rand_chacha::ChaCha12Rng,
}
```

### PyO3 boundary (`src/lib.rs`)

```rust
#![forbid(unsafe_code)]

use pyo3::prelude::*;

#[pymodule]
fn _lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpreter, m)?)?;
    m.add_function(wrap_pyfunction!(generate_planets_for_seed, m)?)?;
    Ok(())
}

#[pyfunction]
fn interpreter<'py>(
    py: Python<'py>,
    state: Bound<'py, PyList>,
    env: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let mut ows = py.allow_threads(|| pybind::pylist_to_state(&state, &env))?;
    py.allow_threads(|| interpreter::step(&mut ows));
    pybind::write_state_back(py, &state, &ows)?;
    Ok(state)
}
```

`py.allow_threads` で重い計算中は GIL を解放し、`multiprocessing.Pool` 並列 self-play との相性を確保。

### Python facade (`simulator/rust/python/orbit_wars_rust/__init__.py`)

```python
"""Rust-backed orbit_wars interpreter; auto-registers on import."""
from __future__ import annotations
import os
from kaggle_environments import register

from . import _facade

# import side-effect: 公式 env name "orbit_wars" を差し替える
register("orbit_wars", _facade.environment_dict())
```

`_facade.environment_dict()` が返す dict は公式の `orbit_wars` registry と同じ構造（`interpreter`, `renderer`, `html_renderer`, `specification`, `agents`）。`interpreter` キーだけ差し替え、ほかは `simulator/python/orbit_wars_vendor/orbit_wars.py` から流用。

```python
# _facade.py
import os
from orbit_wars_vendor.orbit_wars import (
    interpreter as python_interpreter,
    renderer, html_renderer, specification, agents,
)
from orbit_wars_rust._lib import interpreter as rust_interpreter

def interpreter(state, env):
    backend = os.environ.get("ORBIT_WARS_BACKEND", "rust")
    if backend == "python":
        return python_interpreter(state, env)
    return rust_interpreter(state, env)

def environment_dict():
    return {
        "interpreter": interpreter,
        "renderer": renderer,
        "html_renderer": html_renderer,
        "specification": specification,
        "agents": agents,
    }
```

---

## Frontend Design

該当なし（このリポは frontend なし）。`html_renderer` は公式 vendored copy を再利用するのみ。

---

## Data Model Changes

- 既存 `data/` ディレクトリ構造 (lake / processed / mart / output) は変更なし。
- DVC `dvc.yaml` の `cmd: uv run --directory backend` を `bot` に書き換え（path のみ）。
- 新たに **`simulator/` 配下のデータ追跡は不要**（コード + LICENSE のみで data なし）。

---

## Infrastructure Changes

### Dockerfile (`infra/runtime/Dockerfile`)

```dockerfile
WORKDIR /opt/orbit-wars/bot          # 旧 backend
COPY bot/pyproject.toml bot/uv.lock ./
COPY simulator/ /opt/orbit-wars/simulator/    # 新規

# Rust toolchain (Vast.ai 学習ノード用、wheel build 時に必要)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- --default-toolchain stable -y && \
    /root/.cargo/bin/rustup component add clippy rustfmt

ENV PATH="/root/.cargo/bin:$PATH"

# bot を install 時に simulator/rust も maturin develop で焼き込む
RUN uv sync --locked --no-install-project --no-dev && \
    cd /opt/orbit-wars/simulator/rust && \
    /root/.cargo/bin/cargo build --release && \
    uv run maturin develop --release --manifest-path Cargo.toml
```

### CI Workflows

- `.github/workflows/ci-backend.yml` → **`ci-bot.yml`** にリネーム + 内容更新:
  - `uv sync` の前に `actions-rs/toolchain@v1` で Rust toolchain を install。
  - `cd simulator/rust && maturin develop --release` を追加。
  - `cd simulator/rust && cargo fmt --check && cargo clippy -- -D warnings && cargo test` を追加。
  - 既存の `uv run pytest tests` に加え `uv run pytest simulator/python/tests` と `uv run pytest simulator/rust/tests`（parity 用 Python 側）を追加。

- `build-push.yml` の `paths` を `bot/**` と `simulator/**` の両方に拡張。

---

## External Integrations

- **kaggle-environments**: 既存依存を維持。`Environment.run/step/toJSON` は公式実装を使い、interpreter のみ register で差替。
- **PyO3 / maturin**: 新規依存。`simulator/rust/` 直下に閉じる。`bot/pyproject.toml` 側は `orbit-wars-rust` を local path 依存として追加。
- **Rust toolchain (rustup)**: ローカル / CI / Docker で stable channel を install。`dev/setup` の冒頭で `command -v cargo` を確認し、無ければ `rustup` を案内するメッセージを出す（自動 install はしない — risk of unattended modification）。

---

## Build Flow

1. **ローカル初回**:
   ```
   dev/setup  →  uv sync (bot)
              →  cargo build --release (simulator/rust)
              →  uv run maturin develop --release  (venv に orbit_wars_rust._lib を install)
              →  pytest (bot, simulator/python, simulator/rust)
   ```
2. **CI**:
   ```
   ci-bot.yml  →  rustup install stable
               →  uv sync --locked --all-extras --dev (bot)
               →  cd simulator/rust && maturin develop --release
               →  ruff format/lint, mypy, cargo fmt/clippy
               →  pytest (bot/tests + simulator/python/tests)
               →  cargo test (simulator/rust)
   ```
3. **Vast.ai 学習ノード**: Dockerfile に焼き込み済み → `git pull` 後の `uv sync --locked` のみで OK（Rust binary は image layer に含まれる）。

---

## Migration & Compatibility

- **bot/ への rename は単一 PR で git mv**。`backend/` → `bot/` のすべての文字列を一括置換し、CI / Docker / DVC / `.claude/rules/` まで含めてアトミックに更新。
- **simulator/ 配下は新規追加のみ**。既存ファイルへの影響なし。
- **既存 agent コード変更ゼロ** — `import orbit_wars_rust` は `bot/src/agents/__init__.py` または `bot/src/__init__.py` に 1 行追加するだけ（package import 時に side-effect で register 完了）。
- **後方互換不要**: 旧 `backend/` symlink は作らない、旧スクリプト名は残さない（リポ規約）。
