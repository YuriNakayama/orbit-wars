# rulebase-to-jax — Test Strategy

## Testing Approach

`python-to-jax` skill の TDD 規約に従い、**各 port の前に parity test を先書き**する。最重要 NFR は **action 一致率 (parity)**。検証は 3 層: ① 数値 parity (core 単位) → ② action 一致率 (agent 単位) → ③ live 勝率 (実戦健全性)。この順序を厳守し、上流が通るまで下流に進まない。

## Unit Tests

`uv run --directory bot pytest tests/unit/pipeline/rulebase/...`

- **core_jax (Step 1–3)**:
  - `test_core_jax_parity.py`: `geometry/physics/safety` 各関数 vs Python core。`assert_allclose(rtol=1e-5, atol=1e-6)`、int は `array_equal`。
  - `test_worldmodel_jax_parity.py`: 8-turn arrival ledger (planet 別 projected ships) を整数完全一致で検証。lax.scan が jit/vmap 可。
  - `test_missions_jax_parity.py`: score 行列 rtol 1e-5 + 選択 action 完全一致 (tie-break 統一込み)。
- **agent (Step 4–7)**:
  - `test_agent_jax_parity.py` (case1/2/4/5 各々): 大量 obs サンプルで Python `agent(obs)` と JAX `compute_actions_jax(state, seat)` の **action 一致率を測定し 100% を assert**。tie-break ズレを一致率で可視化。
- **opponent (Step 8)**:
  - `test_opponent_jax.py`: strategy_id 配列で 1 関数 vmap が通り、各 id が対応 case の action を返す。

### テストデータ
- obs サンプルは `orbit_wars_jax.reset(seed)` + scripted action sequence で多数局面を生成 (既存 `test_parity.py` 流儀)。noop 盤面・comet spawn (step 50/150/...) を含む seed を網羅。
- tie-break 検証用に score 同値が起きやすい対称局面を意図的に含める。

## Integration Tests

- **Step 9 smoke**: `reinforce/case6` rollout を新 opponent enum で 1-iter 実行し完走確認 (ローカル CPU, episodes 小)。
- e2e trace: 既存 `tests/e2e/pipeline/rulebase/case2/test_agent_jax_identity.py` 流儀で full episode の action 列一致を確認 (case1/2/4/5)。

## E2E / Live Tests (Step 10)

- **GPU throughput bench**: RunPod で vmapped self-play の episode/s を rust/python backend と比較 (`_bench/rulebase_jax_gpu/`)。warm-up 除外 + `block_until_ready` (既存 bench 規約)。
- **live 勝率**: 各 JAX port vs 対応 Python 版 300 戦。**互角 (45–55%) = parity 健全性の最終関所**。乖離時は action 一致率 test に差し戻し。

## Coverage Targets

- Unit: core_jax / agent_jax で 80%+ (repo 既定)。
- Parity: action 一致率 **100%** (tie-break 統一後) を各 case の合格条件とする。
- Live: 各 port vs Python 版 300 戦で 45–55%。
- E2E: case1/2/4/5 の 4 シナリオ + opponent vmap 1 シナリオ。

## CI

`dev/test-bot` (format → lint → mypy → pytest)。GPU bench は CI 外 (RunPod 手動)。case5 rollout 系の重いテストは既存方針通り `slow` マーク検討。
