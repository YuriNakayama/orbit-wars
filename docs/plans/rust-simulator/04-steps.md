# Rust Simulator — Implementation Steps

各ステップは **1 技術関心事 = 1 タスク**の粒度。step 内の work items はさらに parallel に走らせてよい。`Dependencies` は他 step との順序関係を示す。

凡例: `[ ]` = TODO、`backend → bot` の rename は **Step 1 にすべて集約**して以降の step は新パス前提で書く。

---

## Step 1: `backend/` → `bot/` 一括 rename と関連設定の更新 (atomic)

**Target**: cross-cutting (リポ全域)
**Dependencies**: None
**Branch**: 既存 feature/simulator (このプランも同じ branch で良い)

### Overview
リポジトリ全域の `backend` 文字列を `bot` に書き換え、CI/Docker/DVC/`.claude/rules` まで含めた状態でテストが green になることを保証する単一コミット（または極小 PR）。**この PR をマージしてから simulator/ 作業に入る**ことで以降の diff を読みやすくする。

### Work Items
- [ ] `git mv backend bot`
- [ ] dev script を一括更新（並列可）
  - [ ] `dev/setup` の `[ -d "backend" ]` / `cd backend` を `bot` に
  - [ ] `dev/test-backend` を **`dev/test-bot`** にリネーム + 内部 `cd backend`/echo 文言を更新
  - [ ] `dev/format`, `dev/lint`, `dev/submit`, `dev/dvc`, `dev/vast` の `backend` 文字列を `bot` に
  - [ ] `dev/create-worktree` の `ENV_DIRS=("frontend" "backend" "infra")` を `("frontend" "bot" "infra")` に
- [ ] CI workflows を一括更新（並列可）
  - [ ] `.github/workflows/ci-backend.yml` を `ci-bot.yml` にリネーム + 内部 `paths`/`name` を更新
  - [ ] `build-push.yml`: `paths: backend/**` → `bot/**`、`working-directory: backend` → `bot`
  - [ ] `cron-kaggle-scrape.yml`: `working-directory: backend` と `--project backend` / `--directory backend` を `bot` に
- [ ] `infra/runtime/Dockerfile`: `WORKDIR /opt/orbit-wars/backend` と `COPY backend/...` を `bot` に
- [ ] `dvc.yaml`: 全 stage の `cmd: uv run --directory backend` と `deps: backend/...` / `outs: backend/...` を `bot` に
- [ ] `.claude/CLAUDE.md`: folder structure 表 + 説明文 + rule path を `bot` に
- [ ] `.claude/rules/backend/` を **`.claude/rules/bot/`** にリネーム + 内容の `backend/` 文字列を更新
- [ ] `.claude/rules/python.md` / `.claude/rules/command.md` / `.claude/rules/data.md` の `backend` 参照を `bot` に
- [ ] root `README.md` の `backend` 参照を更新
- [ ] `dev/test-bot` を実行し全 lint/type/test が green であることを確認

### Target Files
- `bot/` (旧 `backend/`、git mv のみ)
- `dev/setup`, `dev/test-bot` (旧 `dev/test-backend`), `dev/format`, `dev/lint`, `dev/submit`, `dev/dvc`, `dev/vast`, `dev/create-worktree`
- `.github/workflows/ci-bot.yml` (旧 `ci-backend.yml`), `.github/workflows/build-push.yml`, `.github/workflows/cron-kaggle-scrape.yml`
- `infra/runtime/Dockerfile`
- `dvc.yaml`
- `.claude/CLAUDE.md`, `.claude/rules/bot/` (旧 `.claude/rules/backend/`), `.claude/rules/{python,command,data}.md`
- `README.md`

### Acceptance Criteria
- `dev/test-bot` が green（既存 backend テスト + ruff + mypy）。
- `git grep -n "backend/"` の hit が **意図的な infra terraform "backend" / `backend "s3"` 文脈のみ**になる。
- DVC stage が dry-run で reproducible（既存 outs を invalidate しない）。

---

## Step 2: `simulator/python/` への公式 orbit_wars vendoring

**Target**: simulator/python (新規)
**Dependencies**: Step 1

### Overview
公式 `kaggle_environments/envs/orbit_wars` を `simulator/python/orbit_wars_vendor/` に Apache-2.0 ライセンス遵守で vendoring。pytest が公式 27 cases をそのまま green にする。

### Work Items
- [ ] `simulator/python/` ディレクトリと `pyproject.toml` (hatch, package = `orbit_wars_vendor`) を作成
- [ ] 公式リポから固定 commit (例: master HEAD at 2026-05-02) の `orbit_wars.py` / `orbit_wars.json` / `orbit_wars.js` / `test_orbit_wars.py` / `README.md` を `simulator/python/orbit_wars_vendor/` 配下に **無修正コピー**
- [ ] `simulator/python/LICENSE` に Apache-2.0 LICENSE テキストを置く
- [ ] `simulator/python/NOTICE` に「This subdirectory is a vendored copy of <URL> at commit <SHA>」を記載
- [ ] `simulator/python/orbit_wars_vendor/__init__.py` で `interpreter`, `renderer`, `html_renderer`, `specification`, `agents` を re-export
- [ ] `simulator/python/orbit_wars_vendor/tests/test_orbit_wars.py` を pytest で動くようパスを調整 (元 unittest のまま動くなら無修正)
- [ ] `bot/pyproject.toml` の `dependencies` に `orbit-wars-vendor` を local path で追加 (uv workspace でも可)
- [ ] `dev/test-bot` に `(cd simulator/python && uv run pytest tests)` を追加
- [ ] `simulator/README.md` に Python/Rust の役割分担を記載

### Target Files
- `simulator/python/pyproject.toml`
- `simulator/python/LICENSE`, `simulator/python/NOTICE`, `simulator/python/README.md`
- `simulator/python/orbit_wars_vendor/{__init__.py, orbit_wars.py, orbit_wars.json, orbit_wars.js, README.md}`
- `simulator/python/orbit_wars_vendor/tests/test_orbit_wars.py`
- `simulator/README.md`
- `bot/pyproject.toml`, `bot/uv.lock`
- `dev/test-bot`

### Acceptance Criteria
- `(cd simulator/python && uv run pytest tests)` で公式 27 cases green。
- `from orbit_wars_vendor import interpreter, generate_planets` が動く。
- LICENSE/NOTICE/commit SHA がリポに含まれる。

---

## Step 3: `simulator/rust/` の crate skeleton + maturin 連携

**Target**: simulator/rust (新規)
**Dependencies**: Step 1

### Overview
PyO3 + maturin の hybrid layout の crate と pyproject を作り、`maturin develop --release` で venv に `orbit_wars_rust._lib` をインストールできる状態にする（中身は空 `interpreter` で OK、後続 step で実装）。

### Work Items
- [ ] `simulator/rust/Cargo.toml` (cdylib + pyo3 0.22 + rand_chacha + criterion dev-dep)
- [ ] `simulator/rust/pyproject.toml` (maturin, `python-source = "python"`, `module-name = "orbit_wars_rust._lib"`)
- [ ] `simulator/rust/src/lib.rs` に空の `#[pymodule] fn _lib(_, m) -> PyResult<()> { m.add_function(wrap_pyfunction!(stub_interpreter, m)?)?; Ok(()) }` を置く
- [ ] `simulator/rust/python/orbit_wars_rust/__init__.py` を作成（最初は import + register のみ、interpreter は Python facade に委譲）
- [ ] `simulator/rust/python/orbit_wars_rust/_facade.py` で `os.environ.get("ORBIT_WARS_BACKEND", "rust")` 分岐
- [ ] `simulator/rust/python/orbit_wars_rust/_vendor_bridge.py` で `simulator/python/orbit_wars_vendor` の interpreter/renderer/html_renderer/specification/agents を re-export
- [ ] `simulator/rust/README.md` に build 手順
- [ ] `dev/setup` に `(cd simulator/rust && uv run maturin develop --release)` を追加
- [ ] `dev/lint` / `dev/format` に cargo fmt/clippy を追加
- [ ] `bot/pyproject.toml` の `dependencies` に `orbit-wars-rust` を local path で追加

### Target Files
- `simulator/rust/Cargo.toml`, `simulator/rust/pyproject.toml`, `simulator/rust/README.md`
- `simulator/rust/src/lib.rs`
- `simulator/rust/python/orbit_wars_rust/{__init__.py, _facade.py, _vendor_bridge.py}`
- `dev/setup`, `dev/lint`, `dev/format`
- `bot/pyproject.toml`

### Acceptance Criteria
- `(cd simulator/rust && cargo build --release)` 成功。
- `(cd simulator/rust && uv run maturin develop --release)` 成功し、`uv run --directory bot python -c "import orbit_wars_rust"` が import エラーなし。
- `import orbit_wars_rust` 時に `kaggle_environments.register("orbit_wars", ...)` が呼ばれ、`make("orbit_wars", ...)` で **Python backend (公式 vendored)** がそのまま動く（Rust interpreter はまだ stub）。

---

## Step 4: 公式 27 test cases の Rust port (TDD ゴールデン)

**Target**: simulator/rust/tests
**Dependencies**: Step 3

### Overview
公式 `test_orbit_wars.py` の 27 cases を **cargo test / Rust integration test** として port。実装前に test だけ書き、最初は **expected fail** （unimplemented! を返す）状態にする。

### Work Items
- [ ] `simulator/rust/tests/test_symmetry.rs` ← `test_symmetry`, `test_4_player_initialization`, `test_generate_planets_has_diagonal_orbiting_group`, `test_4p_start_always_static_or_diagonal`
- [ ] `simulator/rust/tests/test_combat.rs` ← `test_combat_*` 9 cases
- [ ] `simulator/rust/tests/test_motion.rs` ← `test_fleet_*` 3 cases (sun, leaving board, surviving)
- [ ] `simulator/rust/tests/test_reward.rs` ← `test_reward_*` 7 cases
- [ ] `simulator/rust/tests/test_helpers.rs` で test fixture (state builder) を共有
- [ ] 各テストが現在は **assert で失敗**することを確認 (`cargo test` が red の状態を許容する flag を付けず、`#[ignore]` を一時的に使う or `#[should_panic]` を使う)
- [ ] `dev/test-bot` に `(cd simulator/rust && cargo test)` を追加（最初は `--no-run` でビルドだけ確認）

### Target Files
- `simulator/rust/tests/{test_symmetry,test_combat,test_motion,test_reward,test_helpers}.rs`
- `dev/test-bot`

### Acceptance Criteria
- `cargo test --no-run` 成功（コンパイルが通る）。
- 27 cases が pending 状態 (`#[ignore]` 付き) で列挙されている。

---

## Step 5: Rust state 構造体 + RNG (`state.rs`, `rng.rs`)

**Target**: simulator/rust/src
**Dependencies**: Step 3

### Overview
`OrbitWarsState` / `Planet` / `Fleet` / `CometGroup` を定義し、ChaCha12 RNG をラップする。

### Work Items
- [ ] `src/state.rs`: 構造体定義 + `OrbitWarsState::new(seed: u64) -> Self`
- [ ] `src/rng.rs`: `ChaCha12Rng` ラッパ + `range_inclusive(low..=high)` / `uniform(low, high)` ヘルパ
- [ ] `src/lib.rs` で modules を pub mod 公開
- [ ] `cargo check` パス

### Target Files
- `simulator/rust/src/{state.rs, rng.rs, lib.rs}`

### Acceptance Criteria
- `cargo build --release` 成功。
- struct の derive(Clone, Debug) が付き、後続モジュールから利用可。

---

## Step 6: `generation.rs` (planet/comet generator) と test green 化

**Target**: simulator/rust/src/generation.rs
**Dependencies**: Step 4, Step 5

### Overview
`generate_planets` / `generate_comet_paths` を Rust に port (公式 Python のロジックを 1:1 で写経)。RNG は ChaCha12（Python `random` 互換は要件外）。Step 4 の symmetry / 4p init 系テストを green にする。

### Work Items
- [ ] `src/generation.rs::generate_planets(rng) -> Vec<Planet>` (公式 Phase 1/1.5/2 のロジック)
- [ ] `src/generation.rs::generate_comet_paths(rng, ...) -> Option<Vec<Vec<[f64;2]>>>`
- [ ] `tests/test_symmetry.rs` の `#[ignore]` を外して green に
- [ ] `tests/test_helpers.rs` で `make_initial_state()` を実装

### Target Files
- `simulator/rust/src/generation.rs`
- `simulator/rust/tests/{test_symmetry,test_helpers}.rs`

### Acceptance Criteria
- `cargo test test_symmetry test_4_player_initialization test_generate_planets_has_diagonal_orbiting_group test_4p_start_always_static_or_diagonal` 全 green。

---

## Step 7: `physics.rs` (fleet motion + planet rotation + collision)

**Target**: simulator/rust/src/physics.rs
**Dependencies**: Step 5

### Overview
`point_to_segment_distance`, `distance`, fleet 移動 (speed scaling), planet rotation (angular_velocity), comet path advance, sweep collision を実装。

### Work Items
- [ ] `src/physics.rs::distance` / `point_to_segment_distance` (公式と数式同一)
- [ ] `OrbitWarsState::advance_fleets` (speed scaling, sun/board/planet collision)
- [ ] `OrbitWarsState::rotate_planets` (initial position から angular_velocity * step)
- [ ] `OrbitWarsState::advance_comets` (path_index 進行 + sweep)
- [ ] `tests/test_motion.rs` の `#[ignore]` を外して green に

### Target Files
- `simulator/rust/src/physics.rs`
- `simulator/rust/tests/test_motion.rs`

### Acceptance Criteria
- `cargo test test_fleet_removed_when_hitting_sun test_fleet_removed_when_leaving_board test_fleet_survives_inside_board` 全 green。

---

## Step 8: `combat.rs` (combat resolution)

**Target**: simulator/rust/src/combat.rs
**Dependencies**: Step 7

### Overview
公式の combat ロジック (top vs second, tie 全滅, defender vs attacker capture) を port。

### Work Items
- [ ] `src/combat.rs::resolve_combats(state)` 関数
- [ ] `tests/test_combat.rs` 9 cases の `#[ignore]` を外して green に

### Target Files
- `simulator/rust/src/combat.rs`
- `simulator/rust/tests/test_combat.rs`

### Acceptance Criteria
- `cargo test test_combat_*` 全 9 cases green。

---

## Step 9: `interpreter.rs` (interpreter step 関数)

**Target**: simulator/rust/src/interpreter.rs
**Dependencies**: Step 6, Step 7, Step 8

### Overview
公式 `interpreter()` の per-turn 順序 (process_moves → production → fleet move → planet move → comet → combat → reward) を統合した `step(&mut OrbitWarsState, actions)` を作る。

### Work Items
- [ ] `src/interpreter.rs::step(state, actions)` で 6 phase を実行
- [ ] termination 判定 (max steps / 1 player alive)
- [ ] reward 算出 (top score / max ships)
- [ ] `tests/test_reward.rs` 7 cases を green に
- [ ] `tests/test_full_episode.rs` で 100 turn の e2e (Python と Rust の同 initial state で final score 比較) を追加

### Target Files
- `simulator/rust/src/interpreter.rs`
- `simulator/rust/tests/{test_reward,test_full_episode}.rs`

### Acceptance Criteria
- `cargo test` で **27 cases + e2e すべて green**。

---

## Step 10: PyO3 bindings (`pybind.rs`, `lib.rs` 完成)

**Target**: simulator/rust/src/{pybind.rs, lib.rs}
**Dependencies**: Step 9

### Overview
PyO3 経由の `interpreter(state, env) -> state` を完成させる。Python の `state[*].observation.{planets, fleets, ...}` を Rust 構造体に load → step → 結果を PyDict に書き戻し。

### Work Items
- [ ] `src/pybind.rs::pylist_to_state` / `write_state_back`
- [ ] `src/pybind.rs::pyaction_to_actions`
- [ ] `src/lib.rs::interpreter` (`#[pyfunction]` + `Python::allow_threads`)
- [ ] panic catch → `PyErr` 変換
- [ ] `cargo test` 維持 (PyO3 layer は Python 側 test で検証)

### Target Files
- `simulator/rust/src/{pybind.rs, lib.rs}`

### Acceptance Criteria
- `maturin develop --release` 後、`uv run python -c "from orbit_wars_rust._lib import interpreter; print(interpreter)"` が動く。

---

## Step 11: Python facade を Rust 経路に切替 + register 自動実行

**Target**: simulator/rust/python/orbit_wars_rust/{__init__.py, _facade.py}
**Dependencies**: Step 10

### Overview
`_facade.interpreter` の rust 分岐を `orbit_wars_rust._lib.interpreter` に接続。`__init__.py` import 時に register 完了。

### Work Items
- [ ] `_facade.py::interpreter` で `os.environ.get("ORBIT_WARS_BACKEND", "rust") == "rust"` のとき `_lib.interpreter` を呼ぶ
- [ ] `__init__.py` で `kaggle_environments.register("orbit_wars", _facade.environment_dict())`
- [ ] `bot/src/agents/__init__.py` (or top-level package) に `import orbit_wars_rust` を 1 行追加（既存 `make` 呼び出しが透過的に Rust 化される）
- [ ] `simulator/rust/python/tests/test_facade.py` で:
  - [ ] `ORBIT_WARS_BACKEND=python` で `make("orbit_wars", ...)` の挙動が公式と一致
  - [ ] `ORBIT_WARS_BACKEND=rust` (default) で同 seed の env が一定 step 内で完走

### Target Files
- `simulator/rust/python/orbit_wars_rust/{__init__.py, _facade.py}`
- `simulator/rust/python/tests/test_facade.py`
- `bot/src/agents/__init__.py` (or `bot/src/__init__.py`)

### Acceptance Criteria
- `uv run --directory bot python -c "import orbit_wars_rust; from kaggle_environments import make; env = make('orbit_wars', configuration={'agents': 2, 'seed': 0}); env.run(['random', 'random']); print(len(env.steps))"` で 100+ step 走る。
- ENV var 切替で公式 / Rust の両方が動く。

---

## Step 12: Python ↔ Rust e2e parity test

**Target**: simulator/rust/python/tests/test_parity.py
**Dependencies**: Step 11

### Overview
**同一初期状態 + 同一 actions** に対し、公式 Python interpreter と Rust interpreter の per-step state hash が一致することを 100 episode で確認。

### Work Items
- [ ] テスト helper: 公式 Python で env を 1 turn 進めて初期 planets/comets を確定 → そのまま Rust state に注入する fixture
- [ ] random / starter / scripted の 3 種類の agent を回し、各 step の `observation.planets` / `fleets` を hash 比較
- [ ] tolerance: f64 演算の cross-platform 誤差を許容するため、`relative tolerance 1e-9` 比較ヘルパを用意
- [ ] 100 episodes × 200 step で完全一致 (or tolerance 内) を assert
- [ ] `dev/test-bot` の `pytest` 対象に含める (slow marker を付け、CI には `not slow` で skip → nightly で run)

### Target Files
- `simulator/rust/python/tests/test_parity.py`
- `dev/test-bot`

### Acceptance Criteria
- `pytest simulator/rust/python/tests/test_parity.py -m slow` で 100 episodes 全 green。

---

## Step 13: Benchmark で 10x 達成検証

**Target**: simulator/rust/benches + simulator/rust/python/tests/test_benchmark.py
**Dependencies**: Step 11

### Overview
criterion で Rust 単体の per-step 時間、pytest-benchmark で Python ↔ Rust の wall-clock を計測し、10x 達成を docs に記録。

### Work Items
- [ ] `simulator/rust/benches/parity.rs` (criterion で 200 step × 2 player × 100 反復)
- [ ] `simulator/rust/python/tests/test_benchmark.py` (pytest-benchmark 経由で `ORBIT_WARS_BACKEND` 切替 + speedup 比較)
- [ ] `pytest-benchmark` を `bot/pyproject.toml` の dev-deps に追加
- [ ] `docs/plans/rust-simulator/benchmark_results.md` に macOS arm64 / Linux x86_64 の実測を貼る
- [ ] speedup_ratio < 10 のとき pytest-benchmark で **warn のみ** (CI は fail しない、現実的な env 依存のため)
- [ ] benchmark コマンドを `dev/lint` から外す (時間がかかる)

### Target Files
- `simulator/rust/benches/parity.rs`
- `simulator/rust/python/tests/test_benchmark.py`
- `bot/pyproject.toml`
- `docs/plans/rust-simulator/benchmark_results.md`

### Acceptance Criteria
- 手動実行で `speedup_ratio >= 10` を確認 + 結果を `benchmark_results.md` に記録。

---

## Step 14: self-play executor の本番組み込み

**Target**: bot/src/dataset/selfplay/executor.py
**Dependencies**: Step 11, Step 12

### Overview
`bot/src/dataset/selfplay/executor.py` の lazy `from kaggle_environments import make` の前に `import orbit_wars_rust` を追加する（または package import で自動 register）。手動でも `dev/sync-data` 系コマンド経由で **default Rust backend が効いた状態**で self-play が走ることを確認。

### Work Items
- [ ] `executor.py:95` の `from kaggle_environments import make` の直前 (or 同モジュールトップ) で `import orbit_wars_rust` を確実に走らせる
- [ ] スモークテスト: 既存 `bot/tests/pipeline/imitation/case1/test_agent_integration.py` などが Rust backend で green
- [ ] self-play 100 マッチを実行し、結果が parquet に保存されることを目視確認 (DVC は触らない)
- [ ] `docs/plans/rust-simulator/06-testing.md` に手動検証手順を記載

### Target Files
- `bot/src/dataset/selfplay/executor.py`
- `bot/tests/pipeline/**` (既存 test を Rust backend で再実行)

### Acceptance Criteria
- 既存 `dev/test-bot` が Rust backend (default) で green。
- 100 マッチ self-play がローカルで 10x 高速化（手動計測）。

---

## Step 15: Docker / CI / dev/setup の Rust toolchain 整備 (final cleanup)

**Target**: infra/runtime/Dockerfile + .github/workflows/ci-bot.yml
**Dependencies**: Step 11

### Overview
Vast.ai 学習ノードと CI で Rust toolchain がプリインストール済みになるようにする。

### Work Items
- [ ] `infra/runtime/Dockerfile` に rustup install + `simulator/rust` の `cargo build --release` + `maturin develop --release` を焼き込む
- [ ] `.github/workflows/ci-bot.yml` に `actions-rust-lang/setup-rust-toolchain@v1` を追加 + `cd simulator/rust && cargo fmt --check && cargo clippy -- -D warnings && cargo test`
- [ ] `.github/workflows/build-push.yml` の `paths` を `["bot/**", "simulator/**"]` に拡張
- [ ] `dev/setup` で rustup が無いとき hint メッセージを出す (自動 install しない)
- [ ] PR description に benchmark 結果リンクを残す

### Target Files
- `infra/runtime/Dockerfile`
- `.github/workflows/ci-bot.yml`, `.github/workflows/build-push.yml`
- `dev/setup`

### Acceptance Criteria
- CI が Rust + Python 両方 green で通る。
- Vast.ai docker image rebuild 後、`dev/vast train` が rust backend で動く。

---

## Cross-cutting Notes

- **後方互換不要**（CLAUDE.md の規約）。旧 `backend/` symlink、旧 `dev/test-backend` alias は作らない。
- **既存 `kaggle_environments` 依存はそのまま**（`bot/pyproject.toml` の `kaggle-environments>=1.17.0` を維持）。register で interpreter を差し替えるだけなので、新バージョンが出ても parity test が破綻検出してくれる。
- **import 順序の罠**: `orbit_wars_rust` を import する前に `make("orbit_wars", ...)` を呼ぶと公式 Python が登録されたままになる。`bot/src/__init__.py` 等で **package import 時に必ず import される位置**に置く。
- **Tip — 並列化**: Step 6/7/8 は state.rs の合意さえ取れれば independent で並列実装可。tests/ も module 別なので fork-join しやすい。
