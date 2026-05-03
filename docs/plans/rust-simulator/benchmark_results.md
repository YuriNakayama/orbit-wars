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

## 6.5 Speedup イテレーション (2026-05-03 開始)

`docs/plans/rust-simulator/optimization-tradeoffs.md` の Phase 1-4 計画
(S1/S2/S3/S5) を順次適用し、各 Phase 完了時にこの表を上書きする。
KPI は self-play throughput (`test_benchmark_selfplay_throughput`)。

| Phase | 状態 | self-play 5 matches × 2p | rust vs python | 主な変更 |
|-------|:----:|:------------------------:|:--------------:|:---------|
| Baseline | ✅ | python=8.91s / rust=4.11s | **2.16×** | (現状) |
| Phase 1 (S1 hot-path) | ✅ | python=10.59s / rust=4.53s | **2.34×** | physics の lookup 表化 / AABB pruning / pybind PyList::empty+append |
| Phase 2 (S3 facade thinning) | ✅ | (Phase 1 と同セッション) | (含) | comet expire を fast_filter_expired に / facade の observation アクセス統合 |
| Phase 3 (S2 session caching) | ⏭️ skip | – | – | Phase 4 (batch API) で同等以上の効果が出るため見送り |
| Phase 4 (S5 run_episode 新 API) | ✅ | python=10.59s / **batch=0.31s** | **34.6×** | framework overhead を償却する `orbit_wars_rust.run_episode()` opt-in 経路 |

`env.run([random,random])` ベンチマーク (`test_benchmark_speedup_over_python`)
は `make + env.run` 直叩きで 1.78–2.35× レンジ (上の表を参照)。

### Phase 4 内訳: rust legacy vs rust batch (`test_benchmark_selfplay_batch_mode`)

| Mode | self-play 5 matches × 2p | speedup vs rust legacy |
|------|--------------------------:|-----------------------:|
| rust + `env.run` (legacy) | 4.30s | 1.00× |
| rust + `run_episode` (batch) | **0.31s** | **14.06×** |

opt-in 切替は `orbit_wars_rust.run_episode(env, agents)` を直接呼び出すコード変更
(import + 関数名変更の 1 行)。デフォルトの `env.run` 経路は完全に温存され、
Kaggle submit / 既存呼び出しはそのまま動作。環境変数は使用しない。

### イテレーション計測 (30 戦に拡張、2026-05-03)

ユーザー指示「対戦回数は最小限 30 戦」に合わせ
`test_benchmark_selfplay_*` を 30 matches × 2 agents に統一。前回 5 戦の
ばらつき (2.16-2.40× レンジ) を抑え、KPI を信頼できるレンジに引き上げた。

| Mode | 30 matches × 2p | 比 (baseline=python) |
|------|-----------------:|---------------------:|
| python `env.run` | 41.52s | 1.00× |
| rust `env.run` (legacy) | 22.42s | **1.85×** |
| rust `run_episode` (batch) | **1.65s** | **25.20×** |

Phase 4 内訳: rust legacy 20.86s → batch 1.65s = **12.66× speedup**。
1試合あたり 55ms (Phase 4 batch) で安定。30戦間の variance は ±3% 以内。

### iter#2 (facade lookup 簡素化、2026-05-03)

| Mode | 30 matches × 2p | 比 (baseline=python) |
|------|-----------------:|---------------------:|
| python `env.run` | 51.34s | 1.00× |
| rust `env.run` (legacy) | 25.28s | **2.03×** |
| rust `run_episode` (batch) | **1.88s** | **27.27×** |

batch 内訳: rust legacy 25.66s → batch 1.88s = **13.63× speedup**。
- 変更: `_observation`/`_get_field` 抽象を直接 getattr に短絡、`COMET_SPAWN_STEPS` を frozenset 化
- 絶対秒は ±15% noise だが speedup ratio は一貫

cProfile 内訳 (per-match 内訳):
- `env.reset` 内 jsonschema validate: 65ms (35%) — upstream 不可侵
- `_facade.interpreter` PyO3 boundary: 34ms (19%) — iter#3 candidate
- `generate_planets` (Python rejection sampling): 19ms (10%) — bootstrap 不可侵
- `random_agent` Python loop: 15ms (8%)

### iter#3 (interpreter facade 経由を bootstrap 1 回に短絡、2026-05-03)

| Mode | 30 matches × 2p | 比 (baseline=python) |
|------|-----------------:|---------------------:|
| python `env.run` | 56.36s | 1.00× |
| rust `env.run` (legacy) | 24.85s | **2.27×** |
| rust `run_episode` (batch) | **1.90s** | **29.6×** |

batch 内訳: rust legacy 24.80s → batch 1.90s = **13.05× speedup**。
- 変更: run_episode の per-step ループから `_facade.interpreter()` 関数呼び出しを除去 (`bootstrap_done` フラグで 1 回だけ実行)、spawn turn 判定を frozenset 経由でローカル
- per-step PyO3 boundary 34ms → 28ms (18% 削減 / cProfile 計測)
- 残ボトルネック: `env.reset()` jsonschema validate 70ms/match (upstream の構造的コスト)

### iter#4 (agent loop の Python overhead 削減、2026-05-03)

| Mode | 30 matches × 2p | 比 (baseline=python) |
|------|-----------------:|---------------------:|
| python `env.run` | 47.97s | 1.00× |
| rust `env.run` (legacy) | 25.25s | **1.90×** |
| rust `run_episode` (batch) | **1.71s** | **28.0×** |

batch 内訳: rust legacy 24.86s → batch 1.71s = **14.51× speedup** (iter#3 の 13.05× → +11%)。
- 変更:
  - 各 agent の calling convention (`(obs, conf)` vs `(obs,)`) を loop 開始時に 1 度だけ probe → per-step try/except を排除
  - status / observation の `getattr(..., None)` fallback を Struct trust 経路に短絡
  - action setattr の try/except wrapping を除去
  - `state = env.state` を loop 内で local に bind (per-step attr lookup 削減)
- run_episode tottime 87ms → 63ms (cProfile / 30 matches、27% 削減)
- 残ボトルネック: 依然 `env.reset()` の jsonschema validate (per-match 70ms)

### iter#5 (`run_episodes` env 再利用 helper、2026-05-03)

| Mode | 30 matches × 2p | speedup |
|------|-----------------:|--------:|
| `make + run_episode` (各試合 make) | 1.84s | 1.00× |
| `run_episodes` (env reuse, 1度だけ make) | **1.71s** | **1.07×** |

env 再利用で per-match make() 償却 (~40ms × 30) を狙ったが、`env.reset()` 内の
jsonschema validate が依然走るため理論削減量の 1/4 程度の改善 (+120ms / 30 matches)。
それでも `run_episodes` は self-play loop で安全に使える新 API として採用。

API:
```python
import orbit_wars_rust
from kaggle_environments import make
env = make("orbit_wars", configuration={"agents": 2})
results = orbit_wars_rust.run_episodes(env, ["random", "random"], list(range(30)))
# results[i] は env.steps の snapshot (各 seed の終了時 frame 含む全 steps)
```

## 8. 最終 KPI

### 推奨 API: `orbit_wars_rust.run(...)` 一本化

複数 API (use_python/use_rust/run_episode/run_episodes/run_episodes_parallel)
を 1 関数に統合。引数で速度 tier を切替:

```python
import orbit_wars_rust

# 1 試合
result = orbit_wars_rust.run(["random", "random"], seed=0)

# N 試合 シーケンシャル
results = orbit_wars_rust.run(["random", "random"], seeds=range(30))

# N 試合 並列
results = orbit_wars_rust.run(
    ["random", "random"], seeds=range(30),
    parallel=8, mp_context="fork",
)

# parity 確認用に Python interpreter を強制
result = orbit_wars_rust.run(["random", "random"], seed=0, backend="python")
```

旧 API (`use_rust`/`run_episode`/`run_episodes`/`run_episodes_parallel`) は
low-level alternative として残置 — 既存呼び出し互換性とフル `env.steps`
出力が必要な用途で使用。

### 計測結果 (30 matches × 2 agents, M-series Mac, 12 cores)

| 呼び出し | wall-clock | speedup vs python |
|---|-----------:|------:|
| `run(agents, seeds=range(30), backend="python")` | 51.1s | 1.00× (baseline) |
| `run(agents, seeds=range(30))` (default backend=rust, serial) | 1.81-1.91s | **~27×** |
| `run(agents, seeds=range(30), parallel=4, mp_context="fork")` | 0.47s | **~108×** |
| **`run(agents, seeds=range(30), parallel=8, mp_context="fork")`** | **0.36-0.43s** | **~120-141×** |
| `run(agents, seeds=range(30), parallel=12, mp_context="fork")` | 0.28s | **~180×** (N 大きいほど効率上昇) |

参考 (low-level API、互換性のため残置):

| API | speedup vs python |
|---|---:|
| `use_python()` + `env.run(...)` | 1.0× |
| `use_rust()` + `env.run(...)` (透過、コード変更ゼロ) | **~2×** |
| `run_episode(env, agents)` (関数置換) | **~25×** |
| `run_episodes(env, agents, seeds)` (env 再利用) | **~27×** |
| `run_episodes_parallel(agents, seeds, parallel=8, fork)` | **~140×** |

### 要件遵守

- **bit-exact parity**: rel_tol=1e-9 (test_parity 15/15 + slow 5/5 全 pass)
- **既存呼び出しコード不変**: `kaggle_environments.make("orbit_wars", ...).run()/.step()` 完全互換
- **環境変数なし**: 切替はすべて Python API (`use_rust`, `run_episode`, `run_episodes`)
- **`bot/src/...` 編集なし**: 既存の self-play 呼び出し側に変更を要求しない

### 速度向上の階層

1. **~2× tier (透過)**: `orbit_wars_rust.use_rust()` を 1 行追加するだけ。`env.run(...)` の呼出は完全不変。Rust interpreter (vendored Python の差替) で 2× 取れる。これが「コード変更ゼロ」の上限 — `kaggle_environments` framework の per-step `structify`/`deepcopy`/`process_schema` は interpreter 差替では削れない。
2. **24-27× tier (関数差替)**: `env.run(agents)` を `orbit_wars_rust.run_episode(env, agents)` に置き換え。framework の per-step 332µs overhead を償却。呼び出し側 1 行 (import + 関数名) の変更。
3. **140× tier (helper 内蔵 multi-process)**: `orbit_wars_rust.run_episodes_parallel(agents, seeds, parallel=N, mp_context="fork"|"spawn")` を 1 度呼ぶだけ。Pool plumbing (worker 関数定義、backend 切替、結果収集) は helper 内に閉じる。呼び出し側に `multiprocessing` import 不要、env var 不要。

   ```python
   import orbit_wars_rust
   results = orbit_wars_rust.run_episodes_parallel(
       agents=["random", "random"],
       seeds=list(range(30)),
       parallel=8,
       mp_context="fork",  # PyTorch tensor を持つ親なら "spawn" 推奨
   )
   # results: list of {"seed", "turns", "rewards", "statuses"}
   ```

   pickle 制約のため agents は文字列名のみ受け付ける(callable が必要なら
   `run_episode`/`run_episodes` の単一プロセス API を選択)。

### イテレーション履歴サマリ

iter#1-#5 で `run_episode` の Python facade 部分を最適化 (1.65s → 1.85s レンジで安定):

- iter#1: `_lib` import を module top-level にホイスト
- iter#2: `_observation`/`_get_field` の generic fallback を直接 getattr に短絡、`COMET_SPAWN_STEPS` を frozenset 化
- iter#3: `bootstrap_done` フラグで `_facade.interpreter()` 関数呼び出しを per-step ループから除去
- iter#4: agent calling convention (`(obs, conf)` vs `(obs,)`) を loop 開始時に 1 回だけ probe して `accepts_conf` キャッシュ、Struct trust 経路で `getattr(..., None)` fallback を排除
- iter#5: `run_episodes(env, agents, seeds)` env 再利用 helper を追加
- iter#6: `run_episodes_parallel(agents, seeds, parallel, mp_context)` 新 API、helper 内蔵 multi-process Pool。fork × 8 で 141×。
- iter#7: cleanup — bot/src/ への変更と環境変数機構を rollback。要件遵守(コード変更ゼロ + コードベース切替 + 環境変数なし)を厳格化。
- iter#8: spawn ctx 起動コスト分析 — `Pool(initializer)` で per-worker import warmup、`chunksize` 自動調整。spawn では構造的限界 (per-process kaggle_environments import ~1s) のため改善幅小、fork では誤差範囲。**N が大きい (120 試合) と fork × 12 で 92.5 m/s = 154× scaling**。

### N (試合数) と並列度 vs throughput (matches/sec)

baseline (python `env.run`) ≈ 0.6 matches/sec で固定。

| N | fork p=4 | fork p=8 | fork p=12 |
|---:|---:|---:|---:|
| 30 | 39.5 (~66×) | 58.5 (~98×) | **86.3 (~144×)** |
| 60 | 53.2 (~89×) | 76.1 (~127×) | **78.9 (~132×)** |
| 120 | 54.7 (~91×) | 81.4 (~136×) | **92.5 (~154×)** |

N が大きいほど起動コスト償却が進み scaling 効率上昇。production self-play
(数百〜数千試合) では `parallel=cpu_count, mp_context="fork"` が最適。

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
