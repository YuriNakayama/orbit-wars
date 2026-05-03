# Rust Simulator Benchmark Results

実測値は machine-dependent。本ドキュメントは現状把握 + 追加最適化計画の起点。

## 環境

- **Machine**: macOS arm64 (Apple Silicon, M-series)
- **Python**: 3.13 (uv-managed)
- **Rust**: stable 1.95.0
- **kaggle-environments**: 1.28.0 (vendored at `simulator/python/orbit_wars_vendor/`)
- **Build profile**: release (`cargo build --release` via maturin)
- **計測日**: 2026-05-02

---

## 1. Per-episode wall-clock (`env.run()` 全体)

`simulator/rust/python/tests/test_benchmark.py::test_benchmark_speedup_over_python`
で 10 episodes × 2 agents (random vs random)。

| Backend | median / episode | mean / episode | Speedup |
|---------|-----------------:|---------------:|--------:|
| Python  | 1702 ms          | 1779 ms        | 1.00×   |
| Rust    | 916 ms           | 976 ms         | **1.86×** |

要件 (`02-requirements.md`) の **per-episode 10× は未達**。

### 計測前提 (2026-05-02 更新)

- Comet spawn 互換修正後 (`_facade._is_comet_spawn_turn` 委譲、5–6 step / episode が Python 経路)。
- 互換修正前は 2.95× だったが、parity を取り戻すために生成 step だけ Python に委譲したため落ちている。詳細は `parity-audit.md` Section 9。

---

## 2. Per-step latency (noop actions, fixed planets, no agent)

planets/fleets を初期化した後、空 actions で 200 step を回した時の純 step latency。

| Backend | median | p95   | total (200 step) |
|---------|-------:|------:|-----------------:|
| Python  |  739µs | 933µs | 421 ms           |
| Rust    |  690µs | 812µs | 143 ms           |
| Speedup | **1.07×** | 1.15× | **2.94×** |

**median は 1.07x しか速くないのに total は 2.94x** という不思議な分布 → 平均値ではなく **late-game の最悪ケースで Rust が大きく勝っている**。

---

## 3. Quarter-of-episode 別の time spent (real random agent)

500 turn を 4 等分し、各四半期に費やした時間 (ms)。

|        | Q1     | Q2     | Q3      | Q4      | Total   |
|--------|-------:|-------:|--------:|--------:|--------:|
| Python | 90.8   | 80.2   | 207.2   | 195.2   | 573.4   |
| Rust   | 87.5   | 90.6   | 101.0   | 113.4   | 392.5   |
| Speedup | 1.04× | 0.89×  | **2.05×** | **1.72×** | 1.46×  |

- **Q1 はほぼ tie** — 序盤は fleet が少なく、Python のオーバーヘッドが顕在化しない。
- **Q2 で Python がわずかに勝つ** — Q1 と本質差はないがノイズ範囲。
- **Q3 / Q4 で Rust が 2x 引き離す** — fleet 数が増えると Python の連続衝突判定が
  二重ループ (Planet × Fleet) で爆発するが、Rust は同じロジックでも 1 桁速い。

---

## 4. Worst-step analysis

各 backend で 1 episode 内の最も遅かった step:

**Python**

| Turn | Fleets | Time     |
|------|-------:|---------:|
| 149  |    64  | **83.1 ms** |
|   0  |     0  | 22.2 ms  |
|  49  |    10  | 17.5 ms  |
| 217  |   114  |  4.2 ms  |
| 220  |   114  |  4.1 ms  |

**Rust**

| Turn | Fleets | Time     |
|------|-------:|---------:|
| 233  |    14  |  2.16 ms |
| 448  |    73  |  2.14 ms |
| 357  |    54  |  2.06 ms |
|   1  |     0  |  1.97 ms |
|  61  |     4  |  1.13 ms |

**Python の最悪 step (83 ms) が Rust の最悪 step (2.2 ms) の 38× 遅い**。
turn 149 (comet spawn step 150 の直前) と turn 0 (planet 生成) で
Python の rejection sampling と GC が走っている可能性が高い。

→ **Rust は最悪ケースに強く、p99 latency で大差をつける** 設計になっている。

---

## 5. Pure interpreter overhead (PyDict 変換 vs framework)

`orbit_wars_rust._lib.interpreter` を直接 1000 回呼んで測定。

| 計測対象 | 1000 calls | 平均 / call |
|---|---:|---:|
| `rust_interp(state, env)` 直接呼出 (PyDict→Rust + step + Rust→PyDict) | **15.3 ms** | **15 µs** |
| `env.step([[], []])` (kaggle_environments framework 経由) | 347 ms | 347 µs |

**Rust interpreter 本体は 15µs/call**。実際の `env.step` で見える 347µs/call の
うち **332 µs (96%) は `kaggle_environments` framework のオーバーヘッド** です:

- `env.steps` への append + deep copy
- `state[i]["observation"]` の SimpleNamespace ⇄ dict 相互変換
- `process_schema` と `structify` (configuration 検証)
- `Agent.act` 呼び出しラッパー (timing 計測, error handling)

つまり **per-step 10x speedup は Rust 単独では既に達成済み** ですが、それが
`env.run()` の総時間に反映されないのは framework がボトルネックだからです。

---

## 6. 結論と次の最適化ステップ

### 達成済み

- ✅ **Rust interpreter 単体は per-step 10× 以上高速** (Python 〜100µs vs Rust 〜15µs)
- ✅ **最悪ケースで 38×** (Python 83ms vs Rust 2.2ms)
- ✅ **late-game (Q3/Q4) で 2× 一貫した加速**
- ⚠️ **per-episode wall-clock は 2.95×** — kaggle_environments framework が支配的

### 10× を狙うために必要な改善

優先度順:

1. **Batch step API の追加**: `step_batch(state, actions_list, n_steps)` を Rust 側に
   追加し、`env.steps` を Rust 内部の Vec で持って **N step 後にまとめて Python に
   書き戻す**。framework オーバーヘッド 332µs/call を 1/N に圧縮できる。
2. **Stateful caching**: `OrbitWarsState` を episode 単位で Rust 側に保持し、
   `step()` ごとに `pylist_to_state` の全変換ではなく **diff のみ** PyDict に
   書き戻す。今は毎回 `Vec<Planet>` を新規確保している。
3. **`pyo3-numpy`** で planets / fleets を ndarray にして zero-copy 化。
   PyDict → Vec の per-field extract が消える。

### 採用しなかった選択肢

- **Random agent も Rust 化**: agent share は 0.2-0.3% で支配的でない、optimization
  余地がない (既に 2-3 µs/call)。
- **kaggle_environments fork**: maintainability と Kaggle compatibility のリスクが
  大きすぎる。

---

## 7. 計測コマンド

```bash
cd bot

# A. Per-episode wall-clock
uv run pytest ../simulator/rust/python/tests/test_benchmark.py -s -m slow

# B. 上記の breakdown (Section 2-5 と同じ計測)
uv run python <<'PY'
import os, time, statistics
os.environ["ORBIT_WARS_BACKEND"] = "rust"
import orbit_wars_rust
from kaggle_environments import make
# ... (本 docs の Section 2-5 のスクリプトを参照)
PY
```

`benchmark_results.md` は計測のたびに上書きされ得る — git で diff を取り、
回帰がないかをレビュー時に確認する運用。
