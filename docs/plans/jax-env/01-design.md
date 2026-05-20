# JAX Env — Phase 1: Design

> 作成日: 2026-05-18
> 前提: `00-codebase-research.md`
> 後続: `02-implementation.md` (Phase B 実装プラン)

## 確定パラメータ

| 定数 | 値 | 根拠 |
|---|---|---|
| `NUM_AGENTS` | **2 / 4 両対応** | 2 が default、4 は FFA。同コード base、num_agents は pytree shape の一部 |
| `BOARD_SIZE` | 100.0 | 公式 |
| `MAX_PLANETS` | 48 | 10 group × 4 + comet 5 × 4 + 余裕 |
| `MAX_FLEETS` | **512** | 余裕多め、drop 回避優先 (GPU mem 影響は許容) |
| `EPISODE_STEPS` | 500 | 公式 default |
| `MAX_COMETS` | 5 | spawn step 数 |
| `MAX_COMET_PATH_LEN` | 40 | 公式 sim 上限 |
| **Device** | **device-agnostic** | Phase D ベンチで CPU / GPU を選定 |

## モジュール分割

`bot/src/jax_env/` 配下:

```
jax_env/
├── __init__.py          # 公開 API (reset, step, JaxEnv class)
├── constants.py         # MAX_PLANETS / MAX_FLEETS / NUM_AGENTS_DEFAULT 等
├── state.py             # EnvState pytree (planet_xy/owner/ships, fleet_xy/owner/ships/valid_mask, comet_paths, step, terminated)
├── geometry.py          # distance, point_to_segment_distance, swept_pair_hit (pure jax)
├── planet_gen.py        # generate_planets (CPU / numpy + random, 公式 sim と RNG 一致)
├── comet_gen.py         # generate_comet_paths (CPU / numpy + random)
├── reset.py             # reset(seed, num_agents) -> EnvState (CPU 実行、最後に jnp.array 化)
├── step.py              # step(state, actions) -> (state, rewards, terminated, info) (pure jax, jit対象)
├── combat.py            # resolve_combat (segment_sum + sort)
├── observation.py       # state_to_obs (公式 obs.planets/obs.fleets リスト形式に戻す、numpy 経由)
├── parity.py            # compare_states_with_vendor (parity test ヘルパ)
└── tests/               # parity / unit tests
```

## EnvState pytree (固定 shape)

```python
@jax.tree_util.register_pytree_node_class
class EnvState:
    # planet (固定長 MAX_PLANETS)
    planet_id:       jnp.int32[MAX_PLANETS]    # -1 = invalid slot
    planet_owner:    jnp.int32[MAX_PLANETS]    # -1 = neutral or invalid
    planet_xy:       jnp.float32[MAX_PLANETS, 2]
    planet_radius:   jnp.float32[MAX_PLANETS]
    planet_ships:    jnp.int32[MAX_PLANETS]
    planet_prod:     jnp.int32[MAX_PLANETS]
    planet_initial_xy: jnp.float32[MAX_PLANETS, 2]  # 静止/回転判定 + 軌道計算
    planet_valid:    jnp.bool_[MAX_PLANETS]     # in-use フラグ
    planet_is_comet: jnp.bool_[MAX_PLANETS]

    # fleet (固定長 MAX_FLEETS)
    fleet_owner:     jnp.int32[MAX_FLEETS]
    fleet_xy:        jnp.float32[MAX_FLEETS, 2]
    fleet_angle:     jnp.float32[MAX_FLEETS]
    fleet_ships:     jnp.int32[MAX_FLEETS]
    fleet_from_pid:  jnp.int32[MAX_FLEETS]
    fleet_valid:     jnp.bool_[MAX_FLEETS]

    # comet (固定長 MAX_COMETS × 4 quadrant)
    comet_paths:     jnp.float32[MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2]
    comet_planet_ids: jnp.int32[MAX_COMETS, 4]    # 対応 planet id
    comet_path_index: jnp.int32[MAX_COMETS]       # 現在の path 進行 index
    comet_valid:     jnp.bool_[MAX_COMETS]

    # global
    angular_velocity: jnp.float32
    step:            jnp.int32
    next_fleet_id:   jnp.int32                  # 公式 sim は fleet id 持つが JAX 内部は不要、debug 用
    num_agents:      jnp.int32                  # 2 or 4 (定数として持つ、batch 全体で同一)
    terminated:      jnp.bool_
    rewards:         jnp.float32[NUM_AGENTS_MAX]  # NUM_AGENTS_MAX=4 で最大値、 num_agents 未満は無効
```

NUM_AGENTS_MAX = 4 固定 shape にして、num_agents=2 の場合は rewards[2:] を 0 で埋める。

## Step 関数の擬似コード (jit 対象)

```python
@jax.jit
def step(state: EnvState, actions: Action) -> tuple[EnvState, jnp.ndarray, bool, dict]:
    # 1. expired comet 削除 (comet_path_index >= path_len で valid_mask off)
    state = _expire_comets(state)

    # 2. comet spawn (step+1 in COMET_SPAWN_STEPS のとき、reset 時に事前計算した paths を出す)
    #    ※ 棄却サンプリングを step 内で行うのは不可。reset 時に 5 spawn 分の path を全部生成しておく
    state = _activate_comet_if_due(state)

    # 3. fleet launch (各 agent の action を反映、ships >= planet_ships AND ships > 0)
    state = _launch_fleets(state, actions)

    # 4. production
    state = _produce_ships(state)

    # 5. planet 移動先計算 (rotation / comet)
    new_planet_xy = _compute_planet_positions(state)

    # 6. fleet 移動 (新 xy)、(F, P) 全ペア衝突判定
    state, combat_fleets = _move_fleets_and_detect_collisions(state, new_planet_xy)

    # 7. planet 移動コミット
    state = state.replace(planet_xy=new_planet_xy)

    # 8. combat resolution (planet × owner segment_sum)
    state = _resolve_combat(state, combat_fleets)

    # 9. step++, termination, reward
    state = state.replace(step=state.step + 1)
    state, rewards, terminated = _check_termination(state)

    return state, rewards, terminated, {}
```

## Reset 関数 (CPU 実行、jit 対象外)

```python
def reset(seed: int, num_agents: int = 2) -> EnvState:
    # 1. 公式 sim と同じ Python random で planet 生成 (parity 保証)
    py_rng = random.Random(seed)
    angular_velocity = py_rng.uniform(0.025, 0.05)
    planets_raw = generate_planets(py_rng)  # 公式 sim の関数をそのまま import

    # 2. home 割当 (公式 sim と同じロジック)
    home_group = py_rng.randint(0, len(planets_raw) // 4 - 1)
    base = home_group * 4
    # ... (公式 sim の interpreter 初期化部を再現)

    # 3. comet path を 5 spawn 分まとめて事前計算
    #    公式 sim では spawn 時に generate_comet_paths を呼ぶ。
    #    ここでは reset 時に 5 種類 (step=50,150,250,350,450) 全部生成して MAX_COMETS スロットに格納
    #    spawn 失敗 (comet_paths is None) ならスロットを invalid のまま
    comet_data = _precompute_all_comets(planets_raw, angular_velocity, py_rng)

    # 4. numpy → jnp.array に変換、EnvState 構築
    return EnvState(
        planet_id=jnp.array(...),
        ...
        comet_paths=jnp.array(comet_data, dtype=jnp.float32),
        ...
    )
```

**重要**: comet spawn は reset 時に全部事前計算する。step 内では `step == 50/150/250/350/450` のときに該当スロットを activate するだけ (jit 内で完結)。これにより rejection sampling を step から完全に隔離。

## Parity テスト戦略

### Phase C: 一致レベル別検証

| レベル | テスト内容 | 許容差 |
|---|---|---|
| **L0: 初期 state** | 同 seed で reset、planet 全フィールドが公式 sim と一致 | 完全一致 (整数+座標は public sim と同 RNG 経路) |
| **L1: 1 step 後** | 同 action で 1 step 進めた後の state 比較 | 浮動小数: abs_diff < 1e-5, 整数: 完全一致 |
| **L2: 100 step trajectory** | random action 系列で 100 step 進めた末の state | 累積誤差: abs_diff < 1e-3 |
| **L3: 500 step 完走** | baseline_v1 vs baseline_v1 を 500 step、scores と termination が一致 | scores 完全一致 |

### 一致不可能ケースの記録

- MAX_FLEETS=512 超過 (公式 sim は無制限) → 該当 seed をログ
- float32 累積誤差で combat 結果が浮動小数依存になるレアケース → 該当ケース統計

### parity テスト実装方針

```python
# bot/tests/jax_env/test_parity.py
def test_state_parity(seed, num_steps, action_strategy):
    py_state = run_vendor_sim(seed, num_steps, action_strategy)
    jax_state = run_jax_env(seed, num_steps, action_strategy)
    assert_state_close(py_state, jax_state, tol_float=1e-5, tol_int=0)

@pytest.mark.parametrize("seed", range(100))
def test_initial_state(seed):
    test_state_parity(seed, num_steps=0, action_strategy=None)

@pytest.mark.parametrize("seed", range(50))
def test_500_step_random(seed):
    test_state_parity(seed, num_steps=500, action_strategy="random")
```

## ベンチマーク戦略 (Phase D)

`bot/scripts/bench_jax_env.py` を作成、以下を測定:

| ベンチ | 期待値 |
|---|---|
| **Vendor sim (Python)** 1 ep wall-clock | ~2-5s/ep |
| **JAX env CPU** 1 ep (jit 後) | ~0.2-1s/ep |
| **JAX env CPU vmap(16)** | ~1-3s/16ep (= 0.06-0.2s/ep) |
| **JAX env GPU** 1 ep | ~0.5-2s/ep (launch overhead) |
| **JAX env GPU vmap(32)** | ~1-3s/32ep (= 0.03-0.1s/ep) |

iter1 ペース 4.5min/iter = 4 ep/min → 1 ep ~ 17s (current rollout 込み)。
JAX env で 0.06-0.1s/ep が達成できれば **150-280× 高速化**、保守的に **30-50×** 改善で iter1 7h → 8-15 min が射程。

## Rollout 統合 (Phase D)

`bot/pipeline/reinforce/case1/training/rollout.py` を改修:

1. **vmap で 16 env を並列ロールアウト** (or env_pool で並列実行)
2. **agent forward を 16 ep 分まとめて GPU バッチ** (現状 1 ep ずつ forward)
3. **trajectory length が ep ごとに異なる問題**: max_steps=500 で固定 shape ロールアウト、early termination は mask で扱う
4. **observation** は `state_to_obs` で公式 obs 形式 (list of [id, owner, x, y, ...]) に numpy 経由で復元、既存 featurizer に渡す

## 実装順序 (Phase B)

| 順 | モジュール | 依存 | テスト |
|---|---|---|---|
| 1 | `constants.py` | — | none |
| 2 | `state.py` (pytree 定義) | constants | pytree flatten/unflatten 単体 |
| 3 | `geometry.py` (jax pure func) | — | swept_pair_hit を vendor と比較 |
| 4 | `planet_gen.py` (CPU + py random) | — | vendor `generate_planets` と完全一致 |
| 5 | `comet_gen.py` (CPU + py random) | planet_gen | vendor `generate_comet_paths` と完全一致 |
| 6 | `reset.py` (CPU、jnp 化) | planet_gen, comet_gen, state | L0 parity |
| 7 | `combat.py` | state | combat resolution 単体 |
| 8 | `step.py` の構成要素 (`_launch_fleets`, `_produce_ships`, `_compute_planet_positions`, `_move_fleets`, `_resolve_combat`, `_expire_comets`, `_activate_comet`, `_check_termination`) | combat, geometry, state | 各単体 + L1 parity |
| 9 | `step.py` 統合 + jit | 上記全部 | L2 parity (100 step) |
| 10 | `observation.py` (numpy → list 復元) | state | rollout から呼べる形 |
| 11 | parity test suite | 全部 | L0-L3 |
| 12 | benchmark | 全部 | Phase D |

## 想定リスクと対処

| リスク | 確度 | 影響 | 対処 |
|---|---|---|---|
| comet spawn の rejection sampling が reset で済まない (例: spawn step で planet 配置が変化していて事前計算 path が不整合) | 中 | parity 失敗 | comet path は spawn 時の planet 配置で生成、reset 時に「**もし** step N で生成したら何が出るか」を事前計算しても planet が移動済なら結果が違う |
| float32 累積誤差で 500 step 後の状態が乖離 | 高 | scores 一致しない | float64 オプション提供、許容誤差段階的に決定 |
| MAX_FLEETS=512 超過頻度が予想より高い | 低 | drop で parity 失敗 | iter1 rollout で実測必要、超過したら 1024 に拡張 |
| jit compile 時間が長すぎる | 中 | warmup overhead | step 関数を小さく保つ、`jax.jit(step).lower(state).compile()` で事前 compile |

### 最大リスクの再掲

**comet spawn の事前計算問題**:
公式 sim では `step=50` で comet を spawn する時、その時点の planet 配置 (50 step 進行済の orbital position) を使う。
事前計算しようとすると「reset 時点での initial_planets」を使うことになり、orbital planet が動いた後の配置と異なる。
解決策候補:
- (A) generate_comet_paths は initial_planets を使う仕様なので **問題ない** (公式 sim もそうしている、orbital planet の現在位置は別途内部で再計算してチェック)
- (B) もし不整合があれば、reset 時に各 spawn step 用に planet 進行を simulate して事前生成

→ 公式 sim 行 313-321 を読むと `game_step = spawn_step - 1 + k` で内部計算しているので、spawn 時点の planet 位置は generate_comet_paths が自己完結で計算 → **(A) で OK**。

---

## Phase C: PPO Rollout Integration (2026-05-20 追加)

> 動機: 既存 Phase B (env-side parity) は完了済 (590 tests pass)。実 wall-clock 短縮は rollout に統合し vmap で 16 ep 並列実行して初めて出る。featurizer/forward もバッチ化しないと env step だけでは <20% しか短縮しない (cProfile より env step は全体 16%)。

### C-1: env-only spike (1-2 day)

目的: JAX env 統合インフラ確認 + env-step alone での speedup 上限測定。

実装範囲:
- `src/jax_env/observation.py` に `comet_planet_ids`, `comets`, `initial_planets` フィールド追加
- `pipeline/reinforce/case1/training/jax_env.py` を新設: `OrbitWarsEpisode` 互換 API を JAX 内部実装で提供
- `rollout.py` に `use_jax_env: bool` フラグを追加 (default false)
- vmap は **行わず**、まず 1 env で wall-clock 同等性 (CPU 単発) を確認

success criteria:
- 既存 parity test (60-turn fixture) で JAX env 出力が vendor obs と同一 featurizer 出力を生成 (bit-equal)
- 1-ep wall-clock が ±20% 以内 (CPU)

failure exit:
- もし featurizer-input 形式が JAX env から再現不可能なら全体撤回

### C-2: featurizer JAX 化 (~1 週間)

目的: featurizer の planet 41-dim / global 20-dim / template_ctx 40-dim / candidate 14×8 を全て `jnp` ベースで再実装し、`vmap(ep_axis=0)` で 16-ep バッチ化可能にする。

最大の難所:
- 現 featurizer は Python loop + dict + dataclass を多用。JAX に持ち込めない:
  - `defaultdict(list)` で `arrivals_by_slot` を構築 → JAX は dynamic shape 不可 → `jnp.scatter` + fixed-shape buffer に置換
  - template resolution の `min(cands, key=...)` → masked argmin
  - `simulate_planet_timeline` の Python while-loop → `jax.lax.scan` 化
- **BC 重み bit-互換性必須**: float32 順序の違いで微小な drift が出る。許容差を 1e-6 で固定し、それ以上の場合は BC 重みを retrain (cost 数 GPU-h)

success criteria:
- `tests/unit/pipeline/reinforce/case1/test_featurizer_parity.py` を JAX 実装でも pass (許容差 1e-6)
- microbench で featurize 単発が CPU で ±50% 以内 / GPU vmap(16) で 5×+

failure exit:
- bit-parity を諦め、BC 重みを retrain (時間とコストの追加見積もり 1 週間 / GPU 数十 \$)
- それでも GPU vmap で 5× 未満なら A2 中止、別アプローチ (truncated episodes など) に転換

### C-3: rollout + ppo_update batched (~3-5 day)

目的: rollout を `vmap(16)` で書き換え、trajectory を `[ep, step]` 固定 shape tensor 化。`ppo_update` も batched tensor 前提。

実装範囲:
- `rollout.py`: `collect_rollout_jax(model, opponent, episodes_per_iter, ...)` を新設。env_state を `jnp.stack([reset(seed+i) for i in range(N)])` で 16 並列初期化、各 step を `jax.vmap(env_step)` で並列実行
- per-step の policy forward は PyTorch 維持 (numpy 経由で device 跨ぎ。1 forward = 16 ep batch を 1 GPU 呼び出し)
- early-termination は mask で扱う、max_steps=500 で固定 shape
- `ppo.py`: tensor 入力に統一、minibatch sampling も flattened (ep × step) で実施

success criteria:
- iter1 wall-clock が **RunPod GPU で 50s 以下** (= 5×+ speedup vs 270s serial baseline)
- BC 重み + 同 hyperparam で iter1 win-rate が ±2pp 以内 (機能等価性)

failure exit:
- 50s 未達なら原因分析 (forward bottleneck / vmap overhead / mask cost) → 部分採用 or 撤退

### Phase C 全体の進行ルール

各 phase 完了時に **ユーザに進行確認**:
- C-1 完了 → 「JAX env で featurizer-input 再現可能。続けて C-2 に進む?」
- C-2 完了 → 「featurizer JAX 化済、parity 1e-6。続けて C-3 統合に進む?」
- C-3 完了 → 「iter1 = 50s 達成。bench config で full 5-iter 検証する?」

途中で failure exit に該当した場合、即座に報告して中止判断を仰ぐ。

