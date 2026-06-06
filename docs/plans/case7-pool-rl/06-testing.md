# case7 Pool 形式 RL — テスト戦略

## テストアプローチ
in-JAX rollout / host 側 selector / eval 統合 の 3 層を unit で固め、CPU smoke で結合検証。
本物 env 大規模対戦(300戦)は採否判定の最終段(Step6/7)でのみ実施(memory: n<300 は信頼不可)。

## Unit Tests(pytest, `bot/tests/pipeline/reinforce/case7/test_pool_opponents.py`)
- **test_case8_opponent_rollout**: `opponent="baseline_jax_case8"` で 1 game 完走、reward 有限・符号あり。既存 mode 0-6 が不変(回帰)。
- **test_case8_action_shape**: `_baseline_case8_actions(state, seat)` が `(L,3)` を返し vmap 可能。
- **test_exploiter_prob_cap**: exploiter 多数 entry で `select()` の full+case8 合計選択率が `exploiter_prob_cap` 以下(モンテカルロ N=2000)。
- **test_exploiter_redistribute**: cap 超過分が past-self snapshot に再配分される。
- **test_eval_in_jax_winrate**: 既知相手 noop に対し `_eval_in_jax` が win≈1.0、強相手 case8 で 0<win<1。
- **test_best_gate_uses_eval_win**: eval_win が低い後続 iter で best.pt が更新されない(self-play win が高くても)。
- **test_metrics_schema**: metrics row に `eval_win`/`eval_opponent`/`exploiter_sel_rate` が含まれる。
- **test_horizon_assert**: config horizon<500 で明示エラー(memory バグ防止ガード)。

## Integration / Smoke Tests
- **CPU smoke(2 iter)**: `pool_default.yaml` を ep=2/horizon=500/2 iter で完走、best.pt + ckpt + metrics 出力。foreground 実行(memory: background hang)。
- **selector rebuild 整合**: pool 入替後も win_ema が index 整合で carry-over される。

## E2E / 採否判定(Step6/7、loop 外の手動実行)
- CPU ~20min run → 全 ckpt を vs rl_v0 / vs case8 で外部 eval sweep(iter15 方式)。
- 採用時のみ GPU run → `dev/runpod pull` → 本物 env で対戦確認。
- **slow マーク**: 重い parity/対戦テストは `@pytest.mark.slow`(CI 5min timeout 回避、既存方針)。

## テストデータ
- 固定 seed の EnvState(reset(seed=0, num_agents=2))。
- noop / case8 相手は決定論なので fixture 不要、その場生成。

## カバレッジ目標
- Unit: 新規コード(rollout case8 分岐 / selector cap / eval)90%+。
- Smoke: pool_default.yaml の 1 経路完走。
- 外部対戦: 採否判定時のみ(n≥300、loop 内では行わない)。
