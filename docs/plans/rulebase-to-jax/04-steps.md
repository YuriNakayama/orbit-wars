# rulebase-to-jax — Implementation Steps

実装順序方針: **core-first → case1 full parity で基盤確立 → 多様性 case (2,5) → champion case4 → opponent 統合 → GPU bench**。各 port は `python-to-jax` skill の TDD (parity test 先書き) に従う。

依存関係:
```
S1 core_jax(physics/geometry/safety) ─┐
S2 worldmodel_jax (8-turn scan) ──────┼─→ S3 missions_jax ─→ S4 case1 full
                                       │                        │
                                       │      S5 case2 ─────────┤
                                       │      S6 case5 ─────────┤
                                       │      S7 case4 ─────────┘
                                       └────────────────────────→ S8 opponent dispatcher
                                                                    │
                                                              S9 case6 rollout 統合
                                                                    │
                                                              S10 GPU bench / live 勝率検証
```
並列化可能: S5 / S6 / S7 は S3 完了後に並走可。

---

## Step 1: 共有 JAX core (physics / geometry / safety)

**Target**: rulebase
**Dependencies**: None

### Overview
分岐レスな数値 core を JAX 化。既存 `case2/aim_jax.py` の閉形式を流用しつつ汎用化。

### Work Items
- [ ] `case1/baseline_jax/core_jax/geometry_jax.py`: `dist`, `safe_angle_and_distance` (判別式 + `jnp.where`)
- [ ] `core_jax/physics_jax.py`: `fleet_speed`, `predict_planet_position`(is_rotating mask), `estimate_arrival`(aim_jax 流用)
- [ ] `core_jax/safety_jax.py`: `is_trajectory_sun_safe`, intercept tolerance
- [ ] parity test 先書き: `tests/unit/pipeline/rulebase/case1/test_core_jax_parity.py` (Python core と rtol 1e-5)

### Target Files (Expected)
- `bot/pipeline/rulebase/case1/baseline_jax/core_jax/{geometry,physics,safety}_jax.py`
- `bot/tests/unit/pipeline/rulebase/case1/test_core_jax_parity.py`

### Acceptance Criteria
- 各関数が Python core と rtol 1e-5 / int exact で一致。jit + vmap が通る。

---

## Step 2: worldmodel_jax (8-turn 防衛シミュレーション)

**Target**: rulebase
**Dependencies**: Step 1

### Overview
`world_model.py` の 8-turn arrival ledger を固定 shape state で port。最難所。

> ⚠️ ループ実装の選択: HORIZON=8 と短いため、**まず Python unroll (固定8回) で実装**する。Web 調査 ([jax#16611](https://github.com/jax-ml/jax/issues/16611)) で `lax.scan` は GPU で kernel launch overhead により unroll より遅い場合があると判明。scan はコンパイル時間/メモリ削減用。主用途が GPU vmapped self-play のため execution 速度優先。コンパイル時間/メモリが問題化したら Step 10 bench を見て `lax.scan` へ切替判断。

### Work Items
- [ ] `core_jax/worldmodel_jax.py`: `simulate_defense(state, seat) -> arrival_ledger` (Python unroll 固定8回で初版)
- [ ] loop state を固定 shape pytree で設計 (projected_ships `[MAX_PLANETS]`, fleet active mask `[MAX_FLEETS]`)
- [ ] `fleet_target_planet` を判別式 + `jnp.where` で port、segment_sum で planet 別集計
- [ ] parity test 先書き: 同一 obs で ledger が Python 版と一致
- [ ] (Step 10 後) コンパイル時間/メモリが過大なら `lax.scan` 版を追加し bench 比較

### Target Files (Expected)
- `bot/pipeline/rulebase/case1/baseline_jax/core_jax/worldmodel_jax.py`
- `bot/tests/unit/pipeline/rulebase/case1/test_worldmodel_jax_parity.py`

### Acceptance Criteria
- 8-turn ledger (planet 別 projected enemy ships) が Python 版と整数一致。lax.scan で jit/vmap 可。

---

## Step 3: missions_jax (全 mission 並列 score → argmax)

**Target**: rulebase
**Dependencies**: Step 1, Step 2

### Overview
snipe/reinforce/harass/swarm/evacuate の各 mission を vectorized score 関数群に。`lax.switch` 不使用、全並列 score → mask → argmax。tie-break 統一。

### Work Items
- [ ] `core_jax/missions_jax.py`: `score_snipe/score_reinforce/...` 各 vectorized 関数
- [ ] `[MAX_PLANETS, MAX_PLANETS, NUM_MISSIONS]` score 行列 → mask → per-src argmax → action 構成
- [ ] tie-break: index 最小を選ぶ微小ペナルティを Python/JAX 両方で統一
- [ ] parity test 先書き: 候補 score 行列と最終 action が Python 版と一致

### Target Files (Expected)
- `bot/pipeline/rulebase/case1/baseline_jax/core_jax/missions_jax.py`
- `bot/tests/unit/pipeline/rulebase/case1/test_missions_jax_parity.py`

### Acceptance Criteria
- score 行列 rtol 1e-5、選択 action (from_pid/angle/ships) が Python 版と一致 (tie-break 含む)。

---

## Step 4: case1 baseline_jax full parity 格上げ

**Target**: rulebase
**Dependencies**: Step 3

### Overview
既存 lite `case1/baseline_jax/agent_jax.py` を core_jax を使う full parity 版に格上げ。既存 `baseline_jax_full` を core_jax へ吸収するか温存するか判断。

### Work Items
- [ ] `compute_actions_jax(state, seat)` を core_jax (worldmodel + missions) ベースに再実装
- [ ] crash_exploit / evacuation を含む全 mission を統合
- [ ] e2e action 一致率 parity test (大量 obs サンプル, 一致率 100% を目標)
- [ ] `baseline_jax_full` の扱いを decision (吸収して廃止 or 互換維持)

### Target Files (Expected)
- `bot/pipeline/rulebase/case1/baseline_jax/agent_jax.py`
- `bot/tests/unit/pipeline/rulebase/case1/test_agent_jax_parity.py`

### Acceptance Criteria
- 同一 obs 大量サンプルで Python `baseline.agent.agent` と action 一致率 100% (tie-break 統一後)。

---

## Step 5: case2 baseline_jax (harass / swarm / lookahead)

**Target**: rulebase
**Dependencies**: Step 3 (Step 4 と並走可)

### Overview
case2 の harass/swarm/OM/lookahead を core_jax 上に追加 port。既存 `case2/aim_jax.py` を流用。

### Work Items
- [ ] `case2/baseline_jax/agent_jax.py`: case1 core + harass/swarm mission score 追加
- [ ] lookahead (浅い OM) を mask/where で表現
- [ ] parity test: case2 Python 版と action 一致率

### Target Files (Expected)
- `bot/pipeline/rulebase/case2/baseline_jax/agent_jax.py`
- `bot/tests/unit/pipeline/rulebase/case2/test_agent_jax_parity.py`

### Acceptance Criteria
- case2 Python 版と action 一致率 100% (既知の disabled feature COMET_NPV 等は除外条件を揃える)。

---

## Step 6: case5 baseline_jax (LB1224 monolith → 並列 score 群)

**Target**: rulebase
**Dependencies**: Step 3 (Step 4 と並走可)

### Overview
2455 行 monolith `agent_full.py` を「全 mission を vectorized 評価する関数群 + 統合 scorer」へ再構成して port。最大規模。

### Work Items
- [ ] case5 Python strategy を mission 単位に分解整理（読解 + マッピング表作成）
- [ ] `case5/baseline_jax/agent_jax.py`: 各 mission を並列 score 関数で実装
- [ ] core_jax の差分(case5 固有 helper)を case5 配下に追加
- [ ] parity test: case5 Python 版と action 一致率

### Target Files (Expected)
- `bot/pipeline/rulebase/case5/baseline_jax/agent_jax.py`
- `bot/tests/unit/pipeline/rulebase/case5/test_agent_jax_parity.py`

### Acceptance Criteria
- case5 Python 版と action 一致率 100%。LB1224 戦略の主要 mission が再現。

---

## Step 7: case4 baseline_jax (champion + fleet consolidation)

**Target**: rulebase
**Dependencies**: Step 3 (Step 4 と並走可)

### Overview
production champion case4。case3 系 + fleet consolidation を port。internal rollout(case3 由来) の扱いを判断（重ければ近似 or 並列 score 化）。

### Work Items
- [ ] `case4/baseline_jax/agent_jax.py`: core_jax + fleet consolidation mission
- [ ] `predict_target_position_fractional` / `SAFE_INTERCEPT_HALF_STEP` を physics_jax に追加
- [ ] fleet consolidation (in-flight fleet merge 判定) を mask/segment 演算で port
- [ ] parity test: case4 Python 版と action 一致率

### Target Files (Expected)
- `bot/pipeline/rulebase/case4/baseline_jax/agent_jax.py`
- `bot/tests/unit/pipeline/rulebase/case4/test_agent_jax_parity.py`

### Acceptance Criteria
- case4 Python 版と action 一致率 100%。consolidation 挙動が再現。

---

## Step 8: opponent dispatcher (strategy_id 分岐レス共通関数)

**Target**: cross-cutting
**Dependencies**: Step 4, Step 5, Step 6, Step 7

### Overview
全 case の `compute_actions_jax` を strategy_id で切替える分岐レス共通インターフェースに統一。

### Work Items
- [ ] `reinforce/case6/policy/opponent_jax.py`: `apply_opponent(state, seat, strategy_id)`
- [ ] 全戦略を計算し `jnp.where(strategy_id==k, ...)` で選択
- [ ] strategy_id ↔ case の enum 定義
- [ ] vmap over (episode, strategy_id) が通ることを test

### Target Files (Expected)
- `bot/pipeline/reinforce/case6/policy/opponent_jax.py`
- `bot/tests/unit/pipeline/reinforce/case6/test_opponent_jax.py`

### Acceptance Criteria
- strategy_id 配列で異種 opponent を 1 関数 vmap 実行でき、各 id が対応 case の action を返す。

---

## Step 9: reinforce/case6 rollout 統合

**Target**: reinforce
**Dependencies**: Step 8

### Overview
新 port を case6 rollout の opponent enum に登録。

### Work Items
- [ ] `rollout_jax.py` の `OPPONENT_*` に case2/4/5 port を追加
- [ ] config (`kaggle_jax_train.yaml`) に新 opponent 選択肢
- [ ] smoke: 1-iter rollout が新 opponent で走る

### Target Files (Expected)
- `bot/pipeline/reinforce/case6/training/rollout_jax.py`
- `bot/pipeline/reinforce/case6/configs/kaggle_jax_train.yaml`

### Acceptance Criteria
- 新 opponent で 1-iter rollout が完走しエラーなし。

---

## Step 10: GPU throughput bench + live 勝率検証

**Target**: cross-cutting
**Dependencies**: Step 8 (Step 9 と並走可)

### Overview
RunPod GPU で vmapped self-play throughput を計測、各 port を本物 Python 版と 300 戦して互角を確認。

### Work Items
- [ ] `_bench/rulebase_jax_gpu/run_bench.py`: vmapped self-play episode/s を rust/python と比較
- [ ] live 勝率: 各 JAX port vs 対応 Python 版 300 戦 (互角 = parity 健全性確認)
- [ ] RunPod 起動 (`dev/runpod train`) → 結果回収 → 報告

### Target Files (Expected)
- `bot/pipeline/reinforce/_bench/rulebase_jax_gpu/run_bench.py`
- `docs/experiment/rulebase/` 配下に bench / 勝率結果

### Acceptance Criteria
- GPU で vmapped self-play が python backend より有意に高速。各 port vs Python 版 300 戦が互角 (45–55%)。
