# Rust Simulator — Parity Audit Report (2026-05-02)

公式 Python interpreter (`simulator/python/orbit_wars_vendor/orbit_wars.py`) と
Rust port (`simulator/rust/`) の互換性を **網羅的** に検証した結果。

> **2026-05-02 修正済み** (`_facade.py` で comet spawn turn を Python に委譲).
> 検証内容と修正履歴は本書末尾の Section 9 を参照。当初の問題と修正前の
> 検証結果は Section 1–6 にそのまま残してある。

**初回監査時の結論: 互換性に重大な欠落あり**。

---

## 1. 結論サマリ

| 観点 | 状態 |
|------|------|
| API surface (`interpreter(state, env)` 呼出) | ◯ 完全互換 |
| 観測 schema (`obs.planets / fleets / comets / ...`) | ◯ 全フィールド再現 |
| 不正アクションのサニタイズ | ◯ 動作一致 |
| Per-step 物理 (fleet 移動・combat・rotation) | ◯ bit-exact 一致 |
| **Comet spawn at steps 50/150/250/350/450** | ✗ **Rust が一切 spawn しない** |
| **`generate_planets` の初回生成** | △ Python 委譲で回避 (Rust 側に実装なし) |
| **`generate_comet_paths` の rejection sampling** | ✗ 未実装 |
| Reward 計算 (max ships, tie, elimination) | ◯ 一致 |
| Termination 条件 (max steps / 1 player alive) | ◯ 一致 |

**Verdict**: **comet 系メカニクスが完全に消失**。post-hoc parity test が通っていたのは、テスト fixture が **Python 側で生成した comet を Rust に注入していた** からで、Rust 単独実行では **comet 0 個** で全 episode が終わる。

---

## 2. 検証したシナリオと結果

### Scenario A: noop actions (空アクション)
- 2 player × 10 seeds × 100 turns: **完全一致**
- 4 player × 5 seeds × 100 turns: **完全一致**

### Scenario B: random actions
- 2 player × 5 seeds × 200 turns: **完全一致**
- 2 player × 3 seeds × 500 turns (full episode): **完全一致 (reward 含む)**
- 4 player × 3 seeds × 500 turns (full episode): **完全一致 (reward 含む)**

### Scenario C: 不正アクション
全 8 ケース (negative ships, oversized fleet, wrong owner, malformed tuple, type error 等) で **観測が同一**、エラー型も一致。

### Scenario D: 観測フィールド総点検
`planets, fleets, initial_planets, comets, comet_planet_ids, next_fleet_id, step, angular_velocity, player, remainingOverageTime` を 5 seeds × 50 turns ですべて比較 → **全フィールド一致**。

### Scenario E: Comet 生成 (核心テスト) ⚠️
seed=0 で 500 step 走らせた spawn イベント:

| backend | 生成 step | 生成された comet ids |
|---------|----------|---------------------|
| python  | 50, 150, 250, 350, 450 | 各回 4 個 (5 spawn × 4 = 20 total) |
| **rust**| **0 events**           | **(none)** |

- Python: 仕様通り COMET_SPAWN_STEPS の各タイミングで 4 体 1 組の comet 群を生成
- Rust: **一度も spawn しない**

---

## 3. 根本原因

`simulator/rust/src/generation.rs` は宣言だけのスタブ:

```rust
//! Initial planet generation and comet path sampling.
//!
//! Filled in by Step 6.
```

`simulator/rust/src/interpreter.rs::step` は `advance_comets` (既存 comet を path 上で動かす) だけを呼ぶ:

```rust
let expired_comets = advance_comets(state, &mut outcome);
```

**`(step+1) in COMET_SPAWN_STEPS` チェック → `generate_comet_paths(...)` → 新 planet 4 体追加 → comet group 登録** の一連の処理が **欠如**。

加えて `generate_planets` (planet 初期生成、symmetry / 軌道惑星生成) も未実装。
これは `_facade._planets_present()` が False のとき Python interpreter に委譲する
ことで隠蔽されている。

---

## 4. なぜこれまで気付かなかったか

1. **Parity test の fixture バグ**: `simulator/rust/python/tests/test_parity.py::init_rs_from_py` は Python 側で初期化された `comets` / `comet_planet_ids` を Rust state に **注入** してから両者を回している。これだと「Rust が新規に comet を生成できない」事実が test 上では検出されず、advance_comets のみ走ればよかった。
2. **Reward が一致**: comet が両方無い episode は ship 数勝負だけで決着するため、reward が偶然一致することがある。テスト Scenario B / D の 500 turn 比較ではこの理由で「一致」となっていた。
3. **`COMET_SPAWN_STEPS` 定数は state.rs にある**: 一見「使う準備はある」ように見えるが、実際の `interpreter::step` から参照されない。

---

## 5. 影響評価

### self-play / 学習データ品質への影響
- 公式 Kaggle 環境では comet 経由の追加 production / 占領が戦略の重要要素。
- Rust 経路で取った self-play ログは comet が出現しない非公式分布 → **imitation learning データとして公式ゲームと不整合**。
- Vast.ai 学習で生成した model を Kaggle に submit すると、**未知の comet を見せられた agent が誤判断する可能性**。

### 評価の信頼性への影響
- `evaluation/eval_vs_baseline` 経路で Rust simulator を使うと、comet 戦略を持つ agent が一切評価されない。
- 既存 rulebase agent (case4 / case5) は comet 関連の戦略を持つので、勝率が大きく狂う。

### Kaggle submit への影響
- submit 経路は依然 Python (`pure_python`) なので **submit 自体は安全**。
- でも submit 前の agent 比較で Rust を使っていた場合、本番性能と乖離する。

---

## 6. 修正方針 (推奨)

### 案 1: `generate_comet_paths` を Rust に実装する (完全互換)
- ChaCha12 RNG では Python `random` 互換が取れないので、**生成は Python に委譲**する hybrid 拡張で十分。
- `_facade.interpreter` で `(step+1) in COMET_SPAWN_STEPS` のとき **Python interpreter に 1 step だけ delegate**。次 step から Rust に戻る。
- 影響: spawn step (5 step / episode) だけ Python overhead が乗るが、497/500 step は Rust 経路。
- 実装コスト: 小 (facade 側 ~10 行の追加)。
- 互換性: 完全。

### 案 2: `generate_comet_paths` を Rust ネイティブ実装
- ChaCha12 で動くが、**Python と異なる comet path** を生成する → bit-exact parity が崩れる。
- Kaggle 公式と同等性を保ちたいなら **NG**。
- 採用するなら Rust 専用 mode として明示し、parity test は別系統にする。

### 案 3: ENV var で Rust backend を一旦無効化
- `ORBIT_WARS_BACKEND=python` をデフォルトに戻し、本欠陥が修正されるまで Rust を opt-in に。
- 即座にできる対症療法。

### 推奨ロードマップ
- **緊急** (今すぐ): 案 3 (default=python に戻す) でデータ汚染を停止。
- **短期** (1 日): 案 1 を実装。`_facade._spawn_step()` ヘルパを追加し、5 spawn step だけ Python に投げる。
- **中期**: parity test の fixture から `init_rs_from_py` での comet 注入を **削除** し、純粋に Rust 側で生成された comet と Python 側の comet を比較する **真の parity test** に置き換える。

---

## 7. 関連する未実装 / 監査外項目

調査の過程で気付いた、互換性に関わる他の懸念:

| 項目 | 状態 | リスク |
|------|------|------|
| `generate_planets` (Rust 実装なし) | 既知 — Python 委譲で回避 | comet と同様に hybrid で OK |
| Rust interpreter 内の `state.step` 進行 | framework が更新するので Rust では参照のみ | OK |
| `expired_comets` の path index 管理 | 実装あり (advance_comets で expire 検出) | comet 自体が生成されないので未検証 |
| `COMET_SPAWN_STEPS` 定数 | state.rs に宣言 → 未参照 | 不要だがリンクしない問題あり |
| `kaggle_environments` v1.28.0 と vendored copy の整合 | 調査済 (commit 同一) | OK |

---

## 8. 計測手順 (再現コマンド)

```python
# Comet spawn 検証 (本 audit の Scenario E)
import os, orbit_wars_rust
from kaggle_environments import make

for backend in ("python", "rust"):
    os.environ["ORBIT_WARS_BACKEND"] = backend
    env = make("orbit_wars", configuration={"seed": 0}); env.reset(2)
    spawn_events = []
    prev = set()
    for t in range(500):
        env.step([[], []])
        if env.done: break
        cids = set(env.steps[-1][0]["observation"].get("comet_planet_ids") or [])
        new = cids - prev
        if new:
            spawn_events.append((env.steps[-1][0]["observation"]["step"], sorted(new)))
        prev = cids
    print(backend, spawn_events)
```

期待出力:
- python: 5 spawn events at step 50/150/250/350/450
- rust (修正前): `[]` ← **欠陥**
- rust (修正後): 5 spawn events 一致

---

## 9. 修正履歴 (2026-05-02 適用)

### 採用した方針 (案 1: Python 委譲 hybrid)

`_facade.interpreter` で **生成系のみ Python に委譲**:

```python
def interpreter(state, env):
    backend = os.environ.get("ORBIT_WARS_BACKEND", "rust").lower()
    if backend == "python":
        return python_interpreter(state, env)
    if not _planets_present(state) or _is_comet_spawn_turn(state):
        return python_interpreter(state, env)  # generation-touching turn
    return _rust_interpreter(state, env)
```

`_is_comet_spawn_turn(state)` は `(obs.step + 1) in COMET_SPAWN_STEPS` を
チェックする (5 turn / 500 = 1% の committee)。

### 修正後の検証結果

**Comet spawn parity** (seed 0/1/7/42 の 4 種、500 step):

| seed | python events | rust events | match |
|------|--------------:|------------:|:-----:|
|   0  | 5 spawn       | 5 spawn     | ✓ ids 完全一致 |
|   1  | 5             | 5           | ✓ |
|   7  | 4 (step=350 は upstream rejection sampling 失敗) | 4 | ✓ |
|  42  | 5             | 5           | ✓ |

**Parity test suite** (`simulator/rust/python/tests/test_parity.py`):

| ケース | 結果 |
|---|:---:|
| `test_parity_noop_short` × 3 seeds | ✓ |
| `test_parity_random_actions_with_first_comet` × 3 seeds | ✓ |
| `test_parity_comet_spawn_events` | ✓ |
| `test_parity_full_episode` × 3 seeds (500 turn each) | ✓ |
| `test_parity_4p_full_episode` × 2 seeds (500 turn 4p) | ✓ |
| **合計** | **12 / 12 pass** |

### Test fixture の改修

修正前の `init_rs_from_py` は **Python 側で生成された comet を Rust に注入**
していたため、Rust の comet 欠落を見逃していた。今は `_seeded_envs` で
**両 backend を独立に立ち上げ、Python global random を `_step_pair` で同期**
する設計に変更。両 backend が **同じ initial state を自前で生成** することが
検証要件に含まれる。

### 性能への影響

修正前: per-episode 2.95× speedup (random vs random, 5 episodes)
修正後: per-episode **1.86×** speedup (10 episodes 平均)

劣化は (a) 1 episode あたり 5–6 turn を Python に委譲する直接コスト、
(b) Python step が late-game で expensive な fleet 衝突判定を走らせるため。

互換性 vs 速度のトレードオフは正解 (互換性確保が優先)。今後の最適化は
`docs/plans/rust-simulator/optimization-tradeoffs.md` の **Phase 2: Batch
step API** で取り戻す予定。
