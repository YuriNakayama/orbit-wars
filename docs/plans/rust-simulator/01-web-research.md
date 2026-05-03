# Rust Simulator — Web Technical Research

ユーザー前提（Step 1 のヒアリング結果）:
- **公式コードをダウンロードしてから進める**（`simulator/python/` に Apache-2.0 サブツリー丸ごと vendoring）。
- **interface は公式と同等**。`make("orbit_wars", configuration={...})` / `env.step(...)` / `env.run(...)` / `env.toJSON()` の API が **既存の agent コード変更ゼロで動く**ことが要件。
- 利用シーンは self-play / 評価 / 学習。Kaggle 提出 agent には Rust binary は持ち込めない（pure Python 制約）。

---

## Official Documentation

### Kaggle/kaggle-environments
- ライセンス: **Apache License 2.0**（[ルート LICENSE](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/LICENSE) で確認済み）。`simulator/python/` 配下にコピーする際は LICENSE/NOTICE を同梱し、出典 URL とリビジョン (commit SHA) を README に明記する。
- `envs/orbit_wars/` 配下のファイル一覧（[GitHub](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars)）:
  - `orbit_wars.py` (806行) — `interpreter`, `generate_planets`, `generate_comet_paths`, `random_agent`, `starter_agent`, `html_renderer`
  - `orbit_wars.json` (97行) — env 仕様
  - `orbit_wars.js` — single-file replay viewer
  - `test_orbit_wars.py` (585行 / 27 cases) — **そのままゴールデンとして利用予定**
  - `README.md`
  - `visualizer/` サブディレクトリ — フル Vite ビルド版の HTML renderer
- `kaggle_environments` パッケージの `core.py` が `Environment.run/step` を担当しており、ここは **再実装不要**（vendor した orbit_wars.py + 既存 site-packages の core.py がそのまま使える）。

### PyO3 + maturin（[公式ガイド](https://pyo3.rs/)、[maturin リポ](https://github.com/PyO3/maturin)）
- **推奨レイアウト**（[maturin user guide](https://www.maturin.rs/tutorial.html)）:
  ```
  my-project/
    Cargo.toml
    pyproject.toml          # [build-system] requires = ["maturin"]
    src/lib.rs              # #[pymodule]
    python/<pkg>/           # オプション: pure Python 部分
  ```
- `[tool.maturin] python-source = "python"` で **混在レイアウト**にできる。`module-name = "<pkg>._lib"` を指定すると Rust 側が pure Python パッケージの内部モジュールとして公開される。
- **GIL リリース**: `Python::allow_threads(|| { ... })` で重い Rust 計算中だけ GIL を手放せる。self-play の並列化と相性が良い。
- **NumPy 連携**: `pyo3-numpy` でゼロコピー交換可能。今回の orbit_wars は NumPy を使っていない（plain list/dict）ので、まずは PyDict/PyList 経由で OK。
- **maturin develop** で local の venv に直接 wheel を build/install できる。`uv sync` 経由でも `maturin` を build dependency に乗せれば OK。

### Cargo workspace + maturin
- maturin は workspace 対応あり（[PR #1123](https://github.com/PyO3/maturin/pull/1123)）。ただし sdist のリビルドで歴史的に問題があった（[Issue #838](https://github.com/PyO3/maturin/issues/838)）。
- 単一 crate のシンプル構成（`simulator/rust/Cargo.toml` 1 つ）が最も摩擦が少ない。
- `simulator/rust/` 自身に `pyproject.toml` を持たせ、`bot/pyproject.toml`（旧 backend）から `dependencies = ["orbit-wars-simulator @ file://../simulator/rust"]` または **maturin develop で venv にインストール**してもらう運用が現実的。

### Rust RNG ([rand book](https://rust-random.github.io/book/guide-seeding.html), [rand-mt](https://docs.rs/mersenne_twister/))
- `StdRng` は **portable でない**（`rand` ドキュメントが明記）。Python `random` 互換を望むなら **避けるべき**。
- 代替: `rand_chacha::ChaCha12Rng`（algorithm 固定）または **Mersenne Twister `mt19937` crate**（Python `random` と同じ MT19937 の 32bit 版）。
- ただし **完全な Python `random` 互換**は `random.uniform` / `random.randint` の wrapper 順序まで一致させる必要があり、現実的には:
  1. 公式 Python の `generate_planets` / `generate_comet_paths` を **そのまま Python 側に残す**（hybrid）
  2. **interpreter の決定論的部分（fleet 動作・combat・planet rotation）だけ Rust 化**して、ランダム生成は Python に委譲
  3. Rust 側の seed は initial_planets/comets が確定した後の "deterministic step" だけを担う
  この戦略なら **MT19937 互換は不要**になる。

---

## Similar OSS Projects

### Project 1 — [Hugging Face tokenizers](https://github.com/huggingface/tokenizers)
- **Relevance**: Python 純正実装を PyO3 経由で Rust 化した代表例。同じ "API 互換を保ちつつ高速化" のパターン。
- **Approach**: pure Python の wrapper class が Rust `_lib` を import して薄く包む。型変換は Rust 側で `PyDict`/`PyList` を構築し、ユーザー側の Python 互換 API を一切変えない。
- **Reusable patterns**:
  - 「Python 互換 facade + Rust core」二層構造
  - 型変換コストを **batch 単位で償却**（per-call ではなく per-episode で wrap/unwrap）
- **Pitfalls**: 細かいセマンティクス差（Unicode 正規化、空文字列の扱い）で test を全部回さないとずれる。**parity test の網羅性が品質決定要因**。

### Project 2 — [pydantic-core](https://github.com/pydantic/pydantic-core)
- **Relevance**: 既存 Python ライブラリを Rust core に置き換え、API は Python 側 wrapper で維持。
- **Approach**: `pyproject.toml` + `Cargo.toml` の hybrid layout。Rust crate は単独で publish 可能（cargo publish）かつ Python wheel としても publish。
- **Reusable patterns**:
  - **wheel に platform-specific バイナリを焼く** → CI で manylinux + macOS arm64 を build matrix
  - **Pythonic な error 型を Rust 側で構築**（`PyErr::new::<PyValueError, _>(msg)`）
- **Pitfalls**: Rust 側のパニックは catch しないと segfault → 必ず `Result<T, PyErr>` に統一。

### Project 3 — [Hungry Geese Rust CFFI (Kaggle precedent)](https://www.kaggle.com/code/jamesmcguigan/hungry-geese-rust-cffi)
- **Relevance**: **Kaggle simulation env を Rust 化した直接の前例**。orbit_wars の前世代の Kaggle simulator env を CFFI 経由で Rust 化した notebook。
- **Approach**: CFFI（PyO3 ではなく）で Rust binary を Python 側に hook。注意点として、**Kaggle 提出ランタイムでは CFFI/PyO3 binary を持ち込めない**ので、self-play 用途の sandbox 限定（今回と同じ立ち位置）。
- **Reusable patterns**: Kaggle simulator の **observation を bytes (msgpack) 経由で Rust に渡し、結果を bytes で受ける** → PyDict 変換コストを最小化したパターンの参考。
- **Lessons**: PyO3 + maturin の方が現代的な選択肢（`uv` build system との親和性、cargo workspace との統合）。

### Pattern Comparison

| Aspect | This project | tokenizers | pydantic-core | Hungry Geese (CFFI) |
|--------|-------------|------------|----------------|---------------------|
| Binding | PyO3 (推奨) | PyO3 | PyO3 | CFFI |
| API 互換性 | **必須** (既存 agent 全部) | 必須 | 必須 | 試作なので緩い |
| Build tool | maturin | maturin | maturin | cargo + cdylib + ctypes |
| Random parity | **重要** | 不要 | 不要 | 不要 |
| 想定 speedup | **10x 以上** | 10-20x | 5-50x | 20x+ (報告済) |

→ **PyO3 + maturin** の hybrid layout、tokenizers の facade パターン、pydantic-core の wheel build CI、を組み合わせる。

---

## Library / Service Selection

### Rust → Python binding tool

| 候補 | Pros | Cons | Maintenance | Recommendation |
|-----|-----|-----|-----|-----|
| **PyO3 + maturin** | uv とシームレス、`#[pymodule]` 一発で公開、GIL 制御が API 化、Polars/pydantic で実績 | 初回 build にツールチェーン要 (rustup) | 活発 (2026/04 時点で v0.28 系) | ⭐ |
| CFFI (`cdylib` + ctypes) | 単純、Python 標準 lib のみで動く | 型ごとに手書き wrapper、null safety なし、PyDict 構築が面倒 | 機能停滞 | ✗ |
| cxx + cbindgen | C++ 互換 | この project は C++ 不要 | 中 | ✗ |
| rustpython | Python 自体を Rust 実装 | speedup は出ない | 中 | ✗ |

**推薦: PyO3 + maturin**。理由は uv ベースの既存 build system との互換、GIL 制御が必要 (multiprocessing self-play との相互作用)、tokenizers/Polars/pydantic-core という大規模実績、aturin が `develop` モードで venv に直接 install できる点。

### Random number generator (Rust 側)

| 候補 | Pros | Cons | Recommendation |
|-----|-----|-----|-----|
| **Python `random` を残す（hybrid）** | Python 互換性 100%、`generate_planets` の rejection sampling も再実装不要 | Rust→Python コールバックのオーバーヘッド | ⭐ |
| `mt19937` crate | Python `random` と同じ MT19937、bit-exact 互換可 | 互換取りに `_randbelow` 等の wrapper を完全再現する必要あり | △ |
| `rand_chacha::ChaCha12Rng` | 高速、portable、modern | Python とは別系列 (互換なし) | ✗ (互換要件あり) |
| `rand::StdRng` | デフォ | **portable not guaranteed** | ✗ |

**推薦: hybrid (Python `random` を残す)**。`generate_planets` / `generate_comet_paths` は Python 側に置き、Rust interpreter には initial state を渡してもらう。これで MT19937 互換議論を回避でき、TDD のゴールデンも安定する。

---

## API / Protocol Research

### kaggle_environments env contract

`kaggle_environments` のレジストリは `register(name, environment_dict)` の形式（`core.py:39-48`）。ユーザー要件「公式と同 interface」を満たすには、Rust core 上に Python facade を被せる以下の戦略が現実的:

```python
# simulator/python/orbit_wars_compat/__init__.py (Python facade)
from kaggle_environments import register
from orbit_wars_rust import interpreter as rust_interpreter   # PyO3 module
from .vendored import generate_planets, html_renderer, ... # 公式 Python の流用

def make_interpreter(state, env):
    # Rust 版を呼ぶか、ENV var で Python 版に fallback
    if os.environ.get("ORBIT_WARS_BACKEND", "rust") == "rust":
        return rust_interpreter(state, env)
    return python_interpreter(state, env)

register("orbit_wars", {
    "interpreter": make_interpreter,
    "renderer": renderer,
    "html_renderer": html_renderer,
    "specification": specification,
    "agents": {...},
})
```

これで `from kaggle_environments import make` の既存呼び出しは **完全にそのまま動く**（同じ env name で register された interpreter が Rust 版に差し替わるだけ）。ENV var で **Python ↔ Rust の切替**を可能にしておけば parity test が書ける。

### env.toJSON() / replay 互換

`env.toJSON()` は `kaggle_environments.core` 側の機能。Rust interpreter が **`state[i].observation` の同じ keys を構築する**限り、replay は同一フォーマットのまま。ここは重要な不変条件として TDD test 化する。

---

## Research Summary

1. **公式 simulator は Apache-2.0**。`simulator/python/` への vendoring は法的に問題なし。LICENSE/NOTICE 同梱と出典 commit SHA の README 記載が必要。
2. **PyO3 + maturin** が binding 戦略の最適解。tokenizers / pydantic-core / Polars と同じパターン。
3. **既存 agent コードの変更ゼロ** 要件は、**`kaggle_environments.register("orbit_wars", ...)` で interpreter を差し替える**facade で達成可能。Python 側に薄い wrapper を置き、Rust 側は純粋な `interpreter(state, env) -> state` の差し替え。
4. **Random parity** は **hybrid 戦略（generate_planets を Python 側に残す）** で回避可。Rust が担うのは決定論的な per-step 物理（fleet 移動、combat、planet rotation、sweep collision）のみ。
5. **TDD ゴールデンソース**は公式 `test_orbit_wars.py` の 27 cases。これを `simulator/rust/tests/` 側に Rust ネイティブテストとして port、さらに `bot/tests/simulator/test_parity.py` で Python ↔ Rust の per-turn 状態 hash 比較。
6. **10倍高速化の現実性**: pure Python interpreter の typical step time はおおよそ 0.5-2 ms（comets / 100+ fleets で重くなる）。Rust 化で per-step 50-200µs 程度が見込める（Hugging Face tokenizers, pydantic で実証済の桁感）。**ボトルネックは PyDict / PyList 変換**。バッチ単位（episode 単位）で wrap/unwrap、または **Rust 内部に状態を保持して `step()` 呼び出しごとに diff のみ返す**設計で達成可能性が高い。
7. **CI 影響**: maturin で wheel を build する step が増える。`dev/setup` に `maturin develop --release` を組み込む案、CI 上は **build キャッシュを使い rust toolchain を per-job install**。最初は手動 `cargo build --release` でも回せる。

Sources:
- [Kaggle/kaggle-environments orbit_wars source](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars)
- [Apache 2.0 LICENSE (kaggle-environments)](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/LICENSE)
- [PyO3 user guide](https://pyo3.rs/)
- [Maturin user guide](https://www.maturin.rs/)
- [Rust Rand book — seeding](https://rust-random.github.io/book/guide-seeding.html)
- [mersenne_twister crate](https://docs.rs/mersenne_twister/)
- [Hungry Geese Rust CFFI (Kaggle precedent)](https://www.kaggle.com/code/jamesmcguigan/hungry-geese-rust-cffi)
- [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) — Python/Rust hybrid pattern
- [pydantic-core](https://github.com/pydantic/pydantic-core) — wheel build / monorepo pattern
- [Polars](https://pola.rs) — PyO3 high-perf precedent
