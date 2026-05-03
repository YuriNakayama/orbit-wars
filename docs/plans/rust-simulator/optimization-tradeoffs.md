# Rust Simulator — Speed-up Tradeoff Analysis

`benchmark_results.md` (2026-05-02) で **per-episode wall-clock 2.95×、Rust 単体 10×、framework オーバーヘッド 332 µs/call (96%)** であることが確定。要件 (`02-requirements.md`) の「per-episode 10×」を達成するために考えられる施策と、それぞれのトレードオフを並べる。

## 1. 現状の time budget breakdown (per env.step)

| 区分 | 時間 | 占有率 | 加速余地 |
|------|-----:|------:|:--------|
| `kaggle_environments` framework (`structify` ×2 + `deepcopy` + state append) | ~280–332 µs | 80–95% | ★★★ |
| Rust interpreter 本体 (PyDict ⇄ Rust + step()) | ~15 µs | 4–5% | ★ |
| Python random_agent + Python loop | ~3 µs | <1% | ☆ |

> **示唆**: もはや **interpreter を速くしても per-episode は速くならない**。framework と Python loop を抜けないと天井に当たる。

---

## 2. 高速化案カタログ

| 案 | 期待 speedup (per-episode) | 実装コスト | 維持性 / 互換性 | 採用判断 |
|----|---------------------------:|:----------:|:-----:|:--------|
| **A**. Batch step API (`step_batch(state, actions, n)`) | 5–10× | 中 | △ — facade 互換崩れる | 推奨 (条件付き) |
| **B**. Stateful caching (Rust 内に state を episode 単位で持つ) | 1.5–3× | 中 | ○ | 推奨 |
| **C**. NumPy (`pyo3-numpy`) で planets/fleets を ndarray | 1.2–1.5× | 中–高 | △ — observation schema 変更 | 保留 |
| **D**. `kaggle_environments` の structify/deepcopy を skip するモンキーパッチ | 3–5× | 低 | ✗ — upstream 依存崩れる | 不採用 |
| **E**. `kaggle_environments` の自前 fork | 5–10× | 高 | ✗ — 大会互換性破壊 | 不採用 |
| **F**. multiprocessing 並列度を上げる (orthogonal) | N×（CPU コア依存） | 低 | ○ | 既存運用に追加可 |
| **G**. agent 自体も Rust 化 | 1.0–1.05×（agent share <1%） | 高 | ✗ | 不採用 |
| **H**. C 拡張 (Cython) で structify / deepcopy を置換 | 1.5–2× | 高 | ✗ | 不採用 |

---

## 3. 各案の詳細

### A. Batch step API (`step_batch`)

**着想**: framework オーバーヘッド 280 µs/call は **`env.step` 1 回ごと** に発生する。N step を Rust 内で連続実行し、最後にだけ Python に書き戻せば **N で割って償却**できる。

**実装イメージ**:
```python
# Python facade
def step_batch(env, actions_seq):  # actions_seq: list of N action lists
    # 1. Hydrate Rust state once
    # 2. Loop in Rust: for actions in actions_seq: step(...)
    # 3. Write back final state + per-step replay frames
    return env_replay
```

**Speedup の試算** (200 step × 5 episodes):
- 現状: 200 step × 332 µs (framework) + 200 × 15 µs (Rust) ≈ 70 ms / episode
- Batch=50: 4 batch × 332 µs (framework) + 200 × 15 µs (Rust) ≈ 4.3 ms / episode → **16x**
- Batch=200: 1 batch × 332 µs + 200 × 15 µs ≈ 3.3 ms / episode → **21x**

**Pros**:
- 最も大きい speedup ポテンシャル。要件「per-episode 10x」を単独達成可能。
- `env.run()` のような **agent action が事前に決まらない** ケースでも、agent も Rust 化すれば適用可。
- self-play / RL training で **rollout buffer に N step まとめて push** する用途にぴったり。

**Cons**:
- **API 互換性が崩れる**: `kaggle_environments.make("orbit_wars").step(action)` 1-step 単位の呼出は速くならない。Step 14 で組み込んだ `selfplay/executor.py` の `env.run()` パスは部分的にしか恩恵なし (env.run の loop 内で個別 step を呼んでいるため、framework は依然走る)。
- 真の恩恵を得るには **`env.run()` 自体を Rust 側 helper で置き換える** (`run_episode_rust(env, agents)`) 必要があり、`agents` の resolve / timing 計測 / replay 保存を facade で再現する手間がかかる。
- Kaggle 大会で submit する agent 経路には影響しない (submit は 1-step ベース)。

**推奨条件**:
- self-play スループット最大化が最重要要件 (まさに plan 02-requirements.md の primary goal) であれば採用すべき。
- 「`make` / `step` API そのままで透過的に高速化したい」要件と矛盾するので、**新 API として共存**させる (既存の facade は残し、`run_episode_fast(env, agents)` を別エントリポイントで提供)。

### B. Stateful caching

**着想**: 毎 step ごとの `pylist_to_state` で **Vec<Planet>, Vec<Fleet> を新規確保** + フィールド全 read している。これを Rust 側に session を持たせ、**diff のみ** 更新する。

**実装イメージ**:
```rust
struct Session { state: OrbitWarsState }  // arena-allocated
static SESSIONS: HashMap<u64, Session>;

#[pyfunction]
fn interpreter_with_session(session_id: u64, state: PyList, env: PyAny) -> PyResult<PyList> {
    let mut s = SESSIONS.get_mut(&session_id).expect("session");
    apply_action_diff(&state, &mut s.state);  // read only action
    step(&mut s.state, ...);
    write_diff_to_state(&state, &s.state);    // write only changed fields
    Ok(state)
}
```

**Speedup の試算**:
- 現在の Rust 単体 15 µs/call の内訳 (推定): 8µs read + 5µs step + 2µs write
- Session 化で read を 1 µs (action のみ) に削れば: 1 + 5 + 2 = 8 µs → 単体 約 2x
- でも env.step() 全体に対する削減は (15-8)/347 ≈ 2% → **per-episode で誤差**

**Pros**:
- API 完全互換。Python 側コード変更なし。
- 副次効果として **cross-platform parity の安定** (allocator が決定論的に)。

**Cons**:
- **per-episode の見かけ speedup は小さい** (2-5%)。
- Session 寿命管理が面倒 (env が GC されたら session も clear しないとリーク)。
  - `weakref.finalize` を Python facade で張る必要あり。
- Rust 側に `static mut HashMap` を持つので `unsafe` または `Lazy<Mutex<HashMap>>` が要る → `#![forbid(unsafe_code)]` を維持するには `Mutex` 経由 (lock コストあり)。
- multiprocessing self-play では各 worker process で session 別管理 → Python 側 session_id 採番ロジックが要る。

**推奨条件**:
- A と組み合わせると効果が乗る (B 単独では弱い)。

### C. `pyo3-numpy` で ndarray 化

**着想**: planets/fleets を `numpy.ndarray` で渡せば PyDict の per-field extract がゼロコピーになる。

**Pros**:
- **Python 側 agent コードも numpy で書ける** → agent 速度も上がる可能性。
- メモリ局所性 (Rust 側で contiguous f64 array) が改善。

**Cons**:
- **observation schema が変わる** — `obs.planets` が list of list ではなく ndarray になる。既存 agent コードが全部 `.tolist()` の挿入か `arr[i, 5]` 書き換えを要する → **agent コード変更ゼロ要件に反する**。
- Kaggle submit 用 agent も同じ obs を見るので、submit パッケージにも numpy 依存追加が必要 (現状でも入っているので OK)。
- `env.toJSON()` 互換性 — replay JSON は list 形式を期待するので、Rust→ndarray→Python list 変換が結局走る。

**推奨条件**:
- agent コード変更を許容できるなら C を A の上に重ねて 1.5x 上乗せ可能。
- 今回のスコープではプラン外。

### D. structify / deepcopy をモンキーパッチで skip

**着想**: `kaggle_environments.core.Environment.step` の deepcopy を no-op に置き換える。

```python
# bot/src/__init__.py
import copy as _copy
from kaggle_environments import core
core.copy = type("c", (), {"deepcopy": lambda x: x})()
core.structify = lambda x: x  # observation already SimpleNamespace
```

**Pros**:
- 即効性 — 1 行で 200 µs 消える。
- Rust と相性が良い (Rust が PyDict を in-place mutate するので deepcopy が不要)。

**Cons**:
- **upstream `kaggle_environments` の不変条件を破る**。observation の不変性に依存する内部処理 (toJSON、log、replay save) が壊れる可能性が高い。
- 大会本番 (Kaggle) は monkey patch 不可 → 「self-play 限定」の運用になる。
- Library 更新で内部実装が変わると突然壊れる。

**推奨条件**:
- 不採用。fragility が割に合わない。

### E. `kaggle_environments` 全体を fork

**着想**: 公式パッケージを vendoring & 改造して structify/deepcopy を消す。

**Pros**:
- 完全コントロール。framework オーバーヘッドをゼロにできる。

**Cons**:
- **Kaggle 大会の submit agent runtime と乖離する** — submit 時の挙動が読めなくなる。
- maintenance burden 大 (公式アップデート毎に rebase)。
- `simulator/python/` の vendoring 範囲が `orbit_wars/` 配下から `kaggle_environments/` 全体に膨れる。

**推奨条件**:
- 不採用。リスクに見合わない。

### F. multiprocessing 並列度を上げる

**着想**: A/B/C と直交。CPU コア数まで `Pool(processes=N)` で self-play 並列実行。

**Pros**:
- 既に `bot/src/dataset/selfplay/runner.py` で実装済み。
- 「同 wall-clock でより多くサンプル」という要件に直接 fit。

**Cons**:
- 上限は CPU コア数 (M-series Mac で 8-12 倍止まり、Linux server で 16-32)。
- メモリ消費が N 倍。

**推奨条件**:
- A/B 相互補完。**A + F で実効 50-100× の self-play スループット**が射程。

### G. agent 自体を Rust 化

**Cons**:
- agent share が 0.2-0.3% (per-step 2-3 µs) なので **per-episode に効かない**。
- random_agent / starter_agent はこれ以上速くしようがない。
- 学習済み agent (PyTorch) は CUDA 制約で Rust 化できない。

**推奨条件**: 不採用。

### H. Cython で structify/deepcopy を置換

PyO3 と同じ effort で D の effect しか得られない。**不採用**。

---

## 4. 推奨ロードマップ

優先度と総合バランスで以下を提案:

### Phase 1 (短期 / 1 週間以内): **B のみ採用**
- 現状の Rust 単体 15 µs を 8 µs 程度に短縮 (1-2× 単体)。
- API 完全互換 → リスクゼロ。
- per-episode wall-clock は 3.0 → 3.2× 程度の改善 (見かけは小さいが、後続 A の時に効く)。

### Phase 2 (中期 / 2-3 週間): **A を新 API として追加**
- `orbit_wars_rust.run_episode(env_config, agents, seed) -> EpisodeResult` を追加。
- self-play executor (`bot/src/dataset/selfplay/executor.py`) の `env.run` を `run_episode` に切替可能にする ENV var (`ORBIT_WARS_RUN_MODE=batch|step`)。
- 期待 speedup: per-episode **10-20×**。要件達成。
- Kaggle submit / 既存 evaluation コードは `step` モードのまま温存 → 互換性破綻ゼロ。

### Phase 3 (長期 / 必要なら): **C で更に詰める**
- agent インターフェイスを numpy 化する coordinated change が必要なので、**imitation/RL training の集中フェーズ**でだけ採用。
- 期待 +1.3× 上乗せ → 総合 **15-30×**。

### Out-of-scope (本イテレーション)
- **D, E, G, H** — リスク・effort 比で割に合わない。

---

## 5. 結論

| 質問 | 答 |
|------|---|
| なぜ要件 10× が達成できていないか | Rust interpreter 本体は既に 10× 達成。`kaggle_environments` framework のオーバーヘッド 280 µs/call が支配的。 |
| 最小コストで 10× へ届くには | **A (batch run_episode API)** を新 API として追加。既存 `make`/`step` 呼出は touch しない。 |
| 「公式と同じ interface」要件と両立できるか | はい — 既存 facade は残し、Rust ネイティブの batch API を **追加** する形で共存。selfplay executor だけ batch 経路に流せば self-play で 10× 享受、Kaggle submit はそのまま。 |
| いつ着手すべきか | Phase 1 (B) は次 PR で。Phase 2 (A) は self-play スループットが律速の実験を回す前。 |
