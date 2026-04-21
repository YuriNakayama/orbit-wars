# case5 (LB 1224 rulebase) — テスト戦略

## アプローチ

1. **agent smoke テスト**: `kaggle-environments` で 1v1 / FFA を完走させ、status 異常を検出
2. **snapshot テスト**: 固定 obs (`tests/.../snapshots/obs_seed0_turn10.json`) に対する action を JSON 保存し、リファクタによる回帰を検出
3. **単体テスト**: 各 mission builder / world_model 検出 / strategy_helpers / timing を独立に検証
4. **notebook 等価性テスト**: notebook と case5 が同一 obs に対して同一 action を返すことを 5-10 シナリオで確認 (Step 19)
5. **自己対戦評価** (テスト枠外、`evaluation/`): case4 vs case5 100 戦の勝率測定

backend.md 準拠: AAA パターン、fixture 使用、mock 最小限、`kaggle_environments.make("orbit_wars")` で実環境再現。

## ユニットテスト

### Backend (`tests/pipeline/rulebase/case5/`)

| ファイル | 対象モジュール | 主要テストケース |
|---------|--------------|----------------|
| `test_baseline_agent.py` | `baseline.agent` | (1) `env.run([agent, agent])` 1v1 完走 (2) FFA 4-player 完走 (3) action shape 検証 (4) snapshot 一致 |
| `test_world_model.py` | `core/world_model.py` | (1) `simulate_planet_timeline` 順序処理 (2) `keep_needed_map` 計算 (3) `exposed_planet_ids` 検出 (4) `blood_in_water_owners` (FFA限定) (5) `enemy_fights_at_neutrals` (6) `stacked_enemy_keep` |
| `test_timing.py` | `core/timing.py` | (1) `Deadline.from_config` の境界 (`actTimeout=0.5` で `deadline = start + 0.41`) (2) `should_skip` 動作 (3) monkeypatch fake clock |
| `test_physics.py` | `core/physics.py` | (1) 静止/軌道/コメット惑星の位置予測 (2) `segment_hits_sun` (3) `aim_with_prediction` |
| `test_strategy_helpers.py` | `strategy_helpers.py` | (1) `build_modes` ahead/behind/finishing 判定 (2) `target_value` (3) `apply_score_modifiers` (各 modifier) (4) `settle_plan` (5) `settle_reinforce_plan` |
| `test_missions_capture.py` | `missions/capture.py` | (1) 中立惑星に対する capture mission 生成 (2) 敵惑星に対する capture |
| `test_missions_snipe.py` | `missions/snipe.py` | (1) 弱い敵惑星 snipe |
| `test_missions_swarm.py` | `missions/swarm.py` | (1) 2源 swarm (2) 3源 swarm 閾値 (3) `THREE_SOURCE_SWARM_ENABLED=False` で 2源のみ |
| `test_missions_rescue.py` | `missions/rescue.py` ★新規 | (1) 落下予測自軍惑星に対する rescue (2) 救援不可能ケースで mission 生成しない |
| `test_missions_recapture.py` | `missions/recapture.py` ★新規 | (1) 直近で失った惑星の検出 (2) frontier 1.08x multiplier |
| `test_missions_crash_exploit.py` | `missions/crash_exploit.py` | (1) 太陽落下予測 fleet の検出 (2) post_crash_delay |
| `test_missions_reinforcement.py` | `missions/reinforcement.py` | (1) 自軍惑星間の補強 |
| `test_movements_evacuation.py` | `movements/evacuation.py` | (1) 太陽落下危険 fleet の退避 |
| `test_movements_rear_guard.py` | `movements/rear_guard.py` | (1) 後方惑星の警備配置 |
| `test_movements_followup.py` | `movements/followup.py` | (1) 連続着弾の補強 |
| `test_movements_proactive.py` | `movements/proactive_defense.py` ★新規 | (1) 複数敵 stack window 検出 (2) `PROACTIVE_DEFENSE_HORIZON` 範囲外は無視 |
| `test_strategy.py` | `strategy.py` | (1) `plan_moves` 全体走行 (2) deadline 切迫時のスキップ |
| `test_license.py` | `baseline/LICENSE` | (1) LICENSE ファイル存在 (2) 冒頭に "Apache License" を含む |

## Integration テスト

| シナリオ | 対象 | アサーション |
|---------|------|------|
| 1v1 完走 | `baseline_v5` × `baseline_v5` (seed 0) | 500 ターン以内に DONE、艦数差 > 0 |
| FFA 完走 | `baseline_v5` × 3 (seed 0, agents=4) | 500 ターン以内に DONE |
| vs random | `baseline_v5` × `random` (seed 0) | DONE、`baseline_v5` が勝者 |
| Snapshot 一致 | 固定 obs (seed 0, turn 10) | 期待 action と完全一致 |

`@pytest.mark.integration` で marker 付与し、CI では default 実行 (case4 と同じ運用)。

## E2E (実環境) テスト

該当なし (バックエンドエージェント、UI なし)。

代替: **Kaggle dry-run** を E2E 相当として位置づける:

```bash
uv run python -m submit submit rulebase/case5 --dry-run --skip-validation -m "case5 dry-run"
```

archive 作成、main.py ロード、`agent({...obs...})` 実行までを通しで確認。

## テストデータ

| データ | 配置 | 用途 |
|-------|------|------|
| `snapshots/obs_seed0_turn10.json` | `tests/.../case5/snapshots/` | snapshot test の入力 obs |
| `snapshots/action_seed0_turn10.json` | `tests/.../case5/snapshots/` | snapshot test の期待 action |
| 各 mission 用 fixture | `tests/.../case5/conftest.py` | 落下予測 obs, exposed planet obs, blood-in-water obs などの構築済 dict |
| notebook 等価性 obs | `tests/.../case5/snapshots/notebook_eq/*.json` | Step 19 で notebook 実行結果と比較する 5-10 サンプル |

## カバレッジ目標

| 階層 | 目標 |
|-----|------|
| `core/timing.py` | 100% (小さいモジュールなので必達) |
| `core/types.py` | 100% |
| `core/world_model.py` | 80% (検出ロジック中心) |
| `core/physics.py` | 75% (純粋関数群、主要パスのみ) |
| `strategy_helpers.py` | 75% |
| `missions/*` | 70% (各 builder の主要パス) |
| `movements/*` | 70% |
| `strategy.py` | 80% (オーケストレーション、smoke で広くカバー) |
| `agent.py` | 80% |
| **全体** | **80% 以上** |

計測コマンド:

```bash
uv run pytest --cov=pipeline.rulebase.case5.baseline tests/pipeline/rulebase/case5/ --cov-report=term-missing
```

## 評価 (テスト枠外、`evaluation/` で実施)

| 評価 | 実施タイミング | 合格基準 |
|------|------------|---------|
| case4 vs case5 100 戦 | Step 19 | case5 勝率 ≥ 45% |
| notebook vs case5 5-10 obs 等価性 | Step 19 | snapshot 完全一致 (浮動小数は `pytest.approx`) |
| ablation (mission on/off) | Step 19+ | 主要 mission off で勝率が下がることを確認 (notebook ロジックの妥当性検証) |
| Turn time profiling | Step 19 | 99 percentile turn time < 0.82s (`SOFT_ACT_DEADLINE` 内) |

## CI 統合

`dev/test-backend` で以下を順次実行 (既存パイプライン):

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest tests/
```

case5 を追加しても既存 `dev/test-backend` がそのまま通ることが Step 17 の合格条件。
