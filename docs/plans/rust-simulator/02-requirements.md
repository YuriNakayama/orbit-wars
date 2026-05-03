# Rust Simulator — Requirements Definition

## Background and Purpose

Kaggle Orbit Wars コンペでは、self-play による対戦データ収集と rulebase / imitation agent の評価を Python の `kaggle_environments.envs.orbit_wars` 上で回している。Python interpreter は per-step ~0.5–2 ms 程度で、100 戦 × 500 step オーダーの実験では実行時間がボトルネックとなり、実験イテレーション速度を抑制している。

そこで本イテレーションでは:

1. リポジトリトップレベルの `backend/` を **`bot/`** にリネームし、ルートに **`simulator/`** を新設する。
2. `simulator/python/` に **公式 `kaggle_environments/envs/orbit_wars` を Apache-2.0 ライセンスで vendoring** する。
3. `simulator/rust/` に **PyO3 + maturin で Rust 版 simulator** を新規作成する。**TDD** で公式 27 test cases 相当を先に作成してから実装する。
4. **`kaggle_environments.register("orbit_wars", ...)` で interpreter を Rust 製に差し替える** facade を導入し、既存 agent コードを 1 行も変えずに高速化を享受する。
5. **公式 Python と Rust の e2e parity** と **10x 以上の高速化** を benchmark で検証する。

これにより、実験イテレーション速度が 1 桁向上し、同時に `kaggle_environments` バージョンアップに対する **回帰検出（parity test）** の基盤も得られる。

## User Stories

- 開発者として、self-play を 10x 高速で回し、同じ wall-clock でより多くのサンプルを集めたい。
- 開発者として、将来の RL training (PPO/A2C 等) で env step が CPU ボトルネックにならないよう、Rust simulator を training loop から呼び出せるようにしたい。
- 開発者として、rulebase agent の 300戦×4p 評価を短時間で回し、case ごとの ablation を素早く決めたい。
- 開発者として、Rust ↔ Python の parity test を CI で回し、公式 API 互換が崩れていないことを常時保証したい。
- 開発者として、`backend/` → `bot/` のリネームに合わせて dev/scripts、CI、Docker、DVC、`.claude/rules/` がすべて整合した状態でリポジトリを使いたい。

## Functional Requirements

1. **トップレベル rename**:
   - `backend/` → `bot/`（git mv）
   - `dev/test-backend` のスクリプト名は **`dev/test-bot`** に rename し、CI workflow / docs の参照も全て更新する（旧名の symlink/alias は残さない — backwards compat 不要）。
   - `dev/setup`, `dev/format`, `dev/lint`, `dev/submit`, `dev/dvc`, `dev/vast`, `dev/create-worktree` の `backend` 文字列を `bot` に書き換え。
   - `.github/workflows/build-push.yml`, `cron-kaggle-scrape.yml`, `ci-backend.yml`（→ `ci-bot.yml`）の `working-directory` / `paths` / `--directory` / `--project` をすべて `bot` に。
   - `infra/runtime/Dockerfile` の `WORKDIR /opt/orbit-wars/backend` と `COPY backend/...` を `bot` に。
   - `dvc.yaml` の `--directory backend` および `deps`/`outs` パス（`backend/pipeline/...`）を `bot/pipeline/...` に。
   - `.claude/CLAUDE.md` / `.claude/rules/python.md` / `.claude/rules/backend/{pipeline,tests}.md`（→ `.claude/rules/bot/{pipeline,tests}.md`）/ `.claude/rules/command.md` / `.claude/rules/data.md` の文言を更新。

2. **`simulator/python/` の vendoring**:
   - `kaggle_environments/envs/orbit_wars/{orbit_wars.py, orbit_wars.json, orbit_wars.js, test_orbit_wars.py, README.md}` を vendoring する。
   - `simulator/python/LICENSE` に Apache-2.0 を同梱、`simulator/python/NOTICE` に出典 (https://github.com/Kaggle/kaggle-environments) と固定した commit SHA を記載。
   - `simulator/python/tests/` には vendored `test_orbit_wars.py` をベースに pytest 互換テストを置き、`pytest simulator/python/tests` で実行可。
   - `pyproject.toml` を `simulator/python/` 直下にも作成（`hatch` build target を `["orbit_wars_vendor"]` 等）。`bot/pyproject.toml` から相対 path 依存で参照、または `uv` の workspace 機能で結合。

3. **`simulator/rust/` の Rust 実装** (PyO3 + maturin, hybrid layout):
   - `simulator/rust/Cargo.toml` + `simulator/rust/pyproject.toml` (maturin) + `simulator/rust/src/lib.rs` (`#[pymodule]`) + `simulator/rust/python/orbit_wars_rust/` (Python facade) + `simulator/rust/tests/` (cargo tests) + `simulator/rust/benches/` (criterion benchmark)。
   - Rust crate 名: `orbit_wars_rust`（snake_case で Python module 名にもなる）。
   - 公開 API（PyO3 経由）:
     - `interpreter(state: PyList, env: PyDict) -> PyList` — 公式 Python `interpreter(state, env) -> state` と同じシグネチャ。
     - `register_orbit_wars()` — `kaggle_environments.register("orbit_wars", {...})` を Python 側で呼ぶための helper（中身は Python facade）。
   - **Rust 全移行** (RNG strategy A): `generate_planets`, `generate_comet_paths`, fleet/planet/comet step、combat resolution、reward 判定、すべて Rust で実装。
   - **RNG**: `rand_chacha::ChaCha12Rng` を `seed_from_u64(seed)` で初期化。Python `random` との bit-exact 互換は **要件外**（公式 RNG 互換は OoS と明記）。
   - **State 管理**: Rust 側で `OrbitWarsState` 構造体に world state を保持し、`step()` ごとに更新後の state から **PyDict diff** を構築して返す。フル obs を毎回構築するより高速かつ Python 側からは透過的に見える。

4. **公式互換 facade**:
   - `simulator/rust/python/orbit_wars_rust/__init__.py` で `kaggle_environments.register("orbit_wars", {"interpreter": ..., "renderer": ..., "html_renderer": ..., "specification": ..., "agents": {"random": ..., "starter": ...}})` を実行する。
   - `interpreter` は ENV var `ORBIT_WARS_BACKEND` を見て `rust`（デフォルト）/`python` を切替える。`python` 指定時は `simulator/python/` の公式コピーを呼ぶ。
   - `renderer` / `html_renderer` / `specification` / `agents` は **公式 Python 実装をそのまま流用**（OoS による）。
   - `bot/` 側は `bot/src/agents/__init__.py` などに `import orbit_wars_rust` を 1 行入れることで register が完了する import-time side effect 設計（既存 `from kaggle_environments import make` 呼び出しはそのまま）。

5. **`bot/` 側 dev script 更新**:
   - `bot/pyproject.toml` の `dependencies` に `orbit-wars-rust @ file://../simulator/rust`（または uv workspace）を追加。
   - `dev/setup` に `(cd simulator/rust && uv run maturin develop --release)` を追加。
   - `dev/test-bot`（旧 `dev/test-backend`）に `(cd simulator/rust && cargo test)` と `(cd simulator/python && uv run pytest tests)` を組み込む。
   - `dev/lint` に Rust 用の `cargo fmt --check` + `cargo clippy -- -D warnings` を追加。

6. **Benchmark**:
   - `simulator/rust/benches/parity.rs` に criterion ベンチを置く（pure Rust per-step time）。
   - `simulator/python/tests/test_benchmark.py` に pytest-benchmark を使い、`make("orbit_wars", ...)` を Python / Rust 両 backend で計測。`speedup_ratio >= 10` を assert。
   - 実測結果は `docs/plans/rust-simulator/benchmark_results.md` に手動更新。

## Non-Functional Requirements

- **Performance**:
  - **per-episode wall-clock を Python 比 10x 以上**（200 step / 2 player の典型シナリオ）。
  - per-step latency は中央値で **100µs 以下**を目標（Rust 単体）。
- **Correctness (parity)**:
  - 公式 Python と Rust の **e2e parity** が **同一初期状態 + 同一 actions** に対し step ごとの state hash が完全一致（100 episodes / 各 200 steps）。
  - **初期状態生成は Python と Rust で別 RNG 系列**なので、parity test は **Python が生成した initial planets/comets を Rust に注入してから step を回す**形で実施。
- **Determinism**:
  - 同一 seed + 同一 actions で **同一 platform 上では bit-exact 同一 trajectory**。
  - Cross-platform (macOS arm64 ↔ Linux x86_64) では `f64` 演算誤差を許容（`relative tolerance 1e-9`）。完全 bit-exact は要求しない。
- **Memory safety**:
  - `unsafe` Rust の使用禁止（`#![forbid(unsafe_code)]` を crate root に）。
  - Rust 側 panic は必ず `PyErr` に変換（`std::panic::catch_unwind` または `Result<T, PyErr>` 統一）。
- **Build portability**:
  - macOS arm64 (M-series) と Linux x86_64 (Vast.ai 学習ノード) 両方で `cargo build --release` + `maturin develop` が通る。
- **License compliance**:
  - `simulator/python/` 配下に Apache-2.0 LICENSE / NOTICE / 出典 commit SHA を同梱。

## Out of Scope

- **Kaggle 提出パッケージ内部への Rust 組み込み** — submit パスは Python のみ。
- **MT19937 互換による Python `random` ビット互換** — ChaCha12 で独立 seed を採用、`generate_planets` の `random.uniform` 呼び出し順を Rust で再現することは要求しない。
- **`html_renderer` / `visualizer/` の Rust 移植** — 公式 JS renderer をそのまま使う。
- **`kaggle_environments` パッケージ自体の `core.py` / `agent.py` / `evaluate()` の Rust 化** — `Environment.run/step/toJSON` は公式に任せ、interpreter のみ差し替え。
- **CI 上での 10x benchmark の自動 pass/fail** — runner 性能差が大きいので、benchmark は手動実行 + `benchmark_results.md` 更新のみ。CI には parity test だけ載せる。
- **後方互換のための旧 `backend/` symlink** — 一切残さない（リポ規約で backwards compat 不要）。

## Glossary

| Term | Description |
|------|-------------|
| `bot/` | rename 後の旧 `backend/`。Python implementation root。 |
| `simulator/python/` | 公式 `kaggle_environments/envs/orbit_wars` の vendored copy（参照実装 / parity ベースライン）。 |
| `simulator/rust/` | PyO3+maturin で作る Rust 実装。Python module 名は `orbit_wars_rust`。 |
| `parity test` | 同一初期状態・同一 actions に対し公式 Python と Rust の per-step state hash が一致することの確認。 |
| `RNG strategy` | Rust 側は `rand_chacha::ChaCha12Rng` 採用、Python `random` とは bit-exact 互換を持たない。 |
| `interpreter facade` | `kaggle_environments.register("orbit_wars", ...)` で interpreter を差し替えるラッパー。`ORBIT_WARS_BACKEND` env var で rust/python 切替。 |
