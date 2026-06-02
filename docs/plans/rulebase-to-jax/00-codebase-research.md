# rulebase-to-jax — Codebase Research

## 調査目的

現状 `bot/pipeline/rulebase/case0–case9` に実装されている rulebase エージェント群を全て JAX 化する方針を検討する。

## Deep Codebase Analysis

### Area 1: rulebase case 群 (`bot/pipeline/rulebase/case0–9`)

- **Files analyzed**: 各 case の `main.py` / `baseline/` / 一部 `baseline_jax*`
- **Current implementation**:
  | case | 概要 | 規模 | JAX port 有無 |
  |------|------|------|---------------|
  | case0 | nearest planet sniper (チュートリアル/archive) | `main.py` 71行 | なし |
  | case1 | sigmaborov LB897 port (snipe/reinforce/crash/swarm/evac) | `baseline/` 3098行 | **`baseline_jax`(65行 lite) + `baseline_jax_full`(243行 full)** |
  | case2 | case1 + harass/swarm/lookahead/OM | baseline 拡張 | `baseline_jax`(experimental) + `aim_jax.py` hot-path port |
  | case3 | case2 + internal rollout (`lookahead/rollout.py` 325行) | baseline 拡張 | なし |
  | case4 | **production champion**: case3 + fleet consolidation。300戦 70.3% | baseline 拡張 | なし |
  | case5 | LB1224 port (Roman Tamrazov)。`agent_full.py` 2455行 単一ファイル | 2455行 | なし |
  | case6 | case4 + STAY judge | baseline 拡張 | なし |
  | case7 | case6 + multi-turn accumulate | baseline 拡張 | なし |
  | case8 | case4 + predict cache 最適化 (挙動等価, turn_p95 -25%) | baseline 拡張 | なし |
  | case9 | case4 + anti-ping-pong | baseline 拡張 | なし |

- **Key interfaces**:
  - Kaggle entrypoint: `case<N>/main.py` → `sys.path.insert(0, cwd); from baseline.agent import agent`
  - `agent(obs) -> list[[from_pid, angle, num_ships], ...]`
  - 共有コア: `case1/baseline/core/{physics,geometry,safety,world_model,config,types}.py`
- **Patterns used**: NamedTuple (`Planet`, `Fleet`)、辞書/オブジェクト両対応の obs パース、純 Python `math` ベースの幾何
- **Coupling & side effects**: case2–9 は case1 の core を再利用しつつ独自に拡張。core への変更は全 case に波及
- **Test coverage**: snapshot test (`tests/unit/pipeline/rulebase/case1/`)、case2 action parity、e2e trace
- **Gaps identified**: case3–9 に JAX port 無し。core ライブラリの JAX 版は未整備（case1 jax は core を再実装している）

### Area 2: 共有 core ライブラリ (`case1/baseline/core/`)

- `physics.py` (242行): `fleet_speed`(log カーブ), `predict_planet_position`(orbit 回転), `estimate_arrival`(intercept) — **hot path**
- `world_model.py` (708行): arrival ledger, 8-turn 防衛シミュレーション, `fleet_target_planet`(ray-circle), 防衛バッファ — **turn 処理の中核, O(F×P)**
- `geometry.py` (75行): `dist`, `safe_angle_and_distance`
- `safety.py` (246行): `is_trajectory_sun_safe`, intercept tolerance
- `config.py` (155行): `HORIZON=8` 等ハイパラ
- **memory 参照**: `project_case4_hot_path` — cProfile で CPU 96% が physics.py、`predict_planet_position` 56M call

### Area 3: JAX simulator (`simulator/jax/orbit_wars_jax/`, 1687行)

- **既に JAX-native**: `EnvState`(固定 shape pytree, MAX_PLANETS=48/MAX_FLEETS=512), `reset`(host-side numpy RNG), `step`(@jit, vmap可), `geometry`(swept collision), `combat`, `observation.state_to_obs`
- vendor parity test 済み (integer 完全一致、float 1.0 tol)
- これが「この repo の良い JAX」のリファレンス実装

### Area 4: 既存 rulebase→JAX port

- `case1/baseline_jax/agent_jax.py`: `compute_actions_jax(state: EnvState, seat) -> jax.Array` (lite, ~80% fidelity)
- `case1/baseline_jax_full/agent_jax_full.py`: orbit prediction + crash exploit + swarm 入り full parity
- **用途**: reinforce/case6 PFSP の curriculum opponent (`OPPONENT_BASELINE_JAX_LITE=1`, `_FULL=2`)
- `case2/baseline_jax/aim_jax.py`: intercept solver の hot-path port（grid vmap で GPU 681×）

### Area 5: adapter (`simulator/adapter/orbit_wars_sim/`)

- `ORBIT_WARS_BACKEND` env var で `python`(vendor) / default `rust` を切替
- **JAX env は adapter 未登録** — reinforce 訓練コードが `orbit_wars_jax` を直 import

### Area 6: parity test / benchmark

- `tests/unit/jax_env/test_parity.py`: integer exact, float soft
- `tests/unit/pipeline/rulebase/case2/test_aim_jax_parity.py`: angle/pos 1e-3, turns/valid exact
- `bot/pipeline/reinforce/_bench/{aim,jax_env,rollout,featurizer,baseline_jax}_gpu/run_bench.py`: RunPod GPU bench

## Technical Constraints (最重要)

1. **JAX は Kaggle submit で逆効果**: Kaggle は GPU 無し・per-turn 1秒予算・jit cold-start 数秒。per-turn 単発呼び出しの rule agent は JAX で **5.6× 遅く**なる (`aim_jax_gpu` bench: batched 24×24 で 681× だが per-call act() は 0.18×)。
2. **JAX が勝つのは throughput (batch)**: vmap で多数 episode / board / candidate を並列計算する offline GPU 用途のみ。
3. **固定 shape + mask 必須**: 可変長 entity は MAX_* padding + `*_valid` mask。Python `int` seed の jit 再 trace で過去にコンパイルキャッシュ 3GB→SIGABRT。
4. **core の JAX 化は EnvState 前提**: dict obs ではなく `orbit_wars_jax.EnvState` を入力にする必要がある（既存 baseline_jax がそうしている）。

## Key Findings Summary

- **「全 case を JAX 化」の素直な解釈（Kaggle submit を JAX に置換）は技術的に逆効果**。JAX 化の真の価値は **offline GPU**: ① reinforce 訓練の高速 vmapped self-play opponent、② 大量 self-play / 評価のバッチ実行。
- 既に case1 に lite/full の 2 種 JAX port があり、reinforce/case6 で opponent として実戦投入済み。これが拡張のテンプレ。
- memory `project_reinforce_case6_live_eval`: train(JAX近似rule)/eval(本物v1) ギャップが課題。**JAX port の parity 精度が RL 進歩の live 転移に直結** → port は単なる速度でなく忠実度が鍵。
- 全 case を一律 full-port するのは過剰。実戦価値（opponent pool 多様性・champion case4 の忠実 port）で優先度を付けるべき。
