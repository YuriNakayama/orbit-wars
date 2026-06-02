# rulebase-to-jax — Architecture Design

## Overall Diagram

```
                       orbit_wars_jax.EnvState  (固定 shape pytree)
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  共有 JAX core (新規)                            │
        │  case1/baseline_jax/core_jax/                   │
        │   ├ physics_jax.py   fleet_speed/predict_pos    │
        │   ├ geometry_jax.py  dist/safe_angle (閉形式)    │
        │   ├ safety_jax.py    sun_safe (判別式+where)     │
        │   ├ worldmodel_jax.py 8-turn lax.scan 防衛 ledger│
        │   └ missions_jax.py  全 mission 並列 score 関数群 │
        └───────────────────────────────────────────────┘
              │            │            │            │
       case1 │     case2  │     case4  │     case5  │   (各 baseline_jax/agent_jax.py)
   compute_actions_jax(state, seat) -> (MAX_LAUNCHES, 3)
              │            │            │            │
              └────────────┴─────┬──────┴────────────┘
                                 ▼
              opponent_jax.py  (新規, 共通 dispatcher)
   apply_opponent(state, seat, strategy_id) -> action
   strategy_id を data で受け分岐レス mask/select で各戦略を選択
                                 │
                                 ▼
        reinforce/case6/training/rollout_jax.py
        OPPONENT_* enum に登録 → vmapped self-play
```

## 配置設計 (各 case 配下 baseline_jax/)

ヒアリング結果に従い、case 毎に `baseline_jax/` を置く既存慣習を踏襲。ただし full parity port で必要な core ロジック (physics/geometry/safety/worldmodel/missions) は **case1 に共有 JAX core パッケージ `core_jax/` を新設**し、case2/4/5 はそこから import する（Python 版が `case1/baseline/core/` を共有しているのと同じ構造を JAX 側に写す）。

```
bot/pipeline/rulebase/
├── case1/
│   ├── baseline_jax/            (既存 lite を full parity に格上げ)
│   │   ├── agent_jax.py         compute_actions_jax(state, seat)
│   │   └── core_jax/            ★新規: 共有 JAX core
│   │       ├── physics_jax.py
│   │       ├── geometry_jax.py
│   │       ├── safety_jax.py
│   │       ├── worldmodel_jax.py
│   │       └── missions_jax.py
│   └── baseline_jax_full/       (既存 full。core_jax へ吸収 or 温存判断)
├── case2/baseline_jax/
│   ├── agent_jax.py             harass/swarm/lookahead を core_jax 上に追加
│   └── aim_jax.py               (既存 hot-path port を流用)
├── case4/baseline_jax/
│   └── agent_jax.py             fleet consolidation 追加版
├── case5/baseline_jax/
│   └── agent_jax.py             LB1224 strategy port (monolith→並列 score 群へ再構成)
└── (opponent dispatcher)
    bot/pipeline/reinforce/case6/policy/opponent_jax.py  ★新規
```

## Core モジュール設計

### physics_jax.py
- `fleet_speed(ships) -> float`: log カーブ。`jnp.where(ships<=1, 1.0, ...)` で分岐レス。
- `predict_planet_position(planet_xy, initial_xy, is_rotating, angular_velocity, turns) -> xy`: 回転は `is_rotating` mask で `jnp.where`。EnvState の `planet_is_rotating` を直利用。
- `estimate_arrival(...) -> (angle, turns, valid)`: 既存 `aim_jax.py` の閉形式 + refinement を流用。

### worldmodel_jax.py (最難所)
- `simulate_defense(state, seat) -> arrival_ledger`: 8-turn 先読みループ。
- **ループ実装は `lax.scan` と Python unroll を bench で選択**:
  - ⚠️ Web 調査 ([jax#16611](https://github.com/jax-ml/jax/issues/16611), [discussion#16106](https://github.com/jax-ml/jax/discussions/16106)): `lax.scan` は**コンパイル時間/メモリを削減**するが、**GPU では unrolled Python loop より遅い場合がある** (iteration 毎に kernel launch、GPU 固有)。
  - 主用途は vmapped self-play (GPU) なので **execution 速度を優先**。HORIZON=8 は短く unroll 有利の可能性大。
  - 方針: まず Python unroll (固定 8 回) で実装 → コンパイル時間/メモリが問題化したら `lax.scan` に切替。Step 10 bench で確定。
- carry/loop state: 固定 shape `(MAX_PLANETS,)` の projected_ships, `(MAX_FLEETS,)` の fleet 位置/active mask。
- per-step: fleet を 1 turn 進め、ray-circle で到達 planet 判定 → ledger に segment_sum で集計。
- `fleet_target_planet`: 判別式ベース、分岐は `jnp.where`。

### missions_jax.py
- 各 mission を `score_<mission>(state, src_idx, tgt_idx) -> score` の vectorized 関数に。
- 全 `(src, tgt, mission_type)` の score を `[MAX_PLANETS, MAX_PLANETS, NUM_MISSIONS]` で算出 → mask → per-src で argmax → action 構成。
- **tie-break 統一**: 同値 score は index 最小を選ぶよう `argmax` 前に微小 index ペナルティを足す等。

### opponent_jax.py (dispatcher)
- `apply_opponent(state, seat, strategy_id) -> action`。
- strategy_id を data で受け、全 case の `compute_actions_jax` を呼んだ結果を `jnp.where(strategy_id==k, action_k, ...)` で選択（分岐レス）。
- ⚠️ 全戦略を毎回計算するコストは PFSP pool 規模(数種)では許容。コスト増が問題化したら JaxMARL 流 group 別 vmap へ退避。

## Data Model

EnvState は既存のまま（変更なし）。新規データ構造は worldmodel の scan carry (内部 pytree) と mission score 行列のみ、いずれも関数ローカル。

## Infrastructure Changes

- 新規依存なし。GPU bench は RunPod (`dev/runpod`)。CUDA は既存 `--group cuda`。
- `_bench/` に rulebase JAX opponent の self-play throughput bench を追加（既存 `baseline_jax_gpu` 拡張）。

## External Integrations

なし。全て repo 内 (orbit_wars_jax / reinforce/case6)。
