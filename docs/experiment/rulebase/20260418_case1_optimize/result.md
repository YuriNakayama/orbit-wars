# 🎯 [case2] case1 最適化結果レポート (2026-04-19)

`feature/optimize-case1` ブランチで実施した case1 最適化検証の全記録。
目的: Kaggle LB 897 の sigmaborov reinforce port (case1 baseline_v1) をさらに改善した case2 baseline_v2 を作り、勝率を上げる。

結論を先に: **300戦で 50.7% (+1.4pp, 非有意)** にとどまり、Prediction / Lookahead 系の改善は全て seed variance に埋没または害。構造的改修が必要という示唆が得られた。

---

## 📋 実施内容サマリー

| Phase | 施策 | 結果 | 採否 |
|---|---|---|---|
| Refactor | `strategy.py` 702→168行、mission/movement 分離、snapshot test 追加 | 17/17 test pass, 機能等価 | 採用 |
| Phase 3-a | Harass mission 追加 (enemy production ≥2 を短期占領して production 奪取) | 100戦 50-51% (+3〜4pp) | **採用 (default ON)** |
| Phase 3-b | `SAFE_INTERCEPT_HALF_STEP` 有効化 (ETA を 0.5 単位で探索) | 100戦 49% (+2pp) | **採用 (default ON)** |
| Phase 3-c | Harass + HALF_STEP 同時 ON | 100戦 57%/55%、**300戦 50.7%** (+1.4pp 非有意) | 採用維持 |
| Phase 3-d | `DYNAMIC_PROACTIVE_HORIZON` 再検証 | 100戦 47% (ベースライン同等) | OFF 維持 |
| Phase 3-e | `COMET_NPV` 再検証 | 100戦 45% (**-6pp 害**) | OFF 維持 |
| Phase 3-f | `FINISHING_TIE_GUARD` 再検証 | 100戦 46% (**-11pp 害**) | OFF 維持 |
| Phase 4-a | OM v2 (phase-gated prediction + weight multiplier) | 100戦 53%/48%、**300戦 50.7%** (+1.4pp 非有意) | 実装温存、OFF |
| Phase 4-b | Lookahead 1-ply self-play (weight=0.6) | 100戦 42%/51% (**-3.5pp 害**、turn_p95 4倍) | 実装温存、OFF |
| Phase 4-c | Lookahead gated (after_step=80, weight=0.3) | 100戦 50% (差無し、turn_p95 0.19s) | 実装温存、OFF |

**最終構成** (case2 baseline_v2 default): `HARASS_ENABLED=True` + `SAFE_INTERCEPT_HALF_STEP=True`、その他の実験的 flag は全て OFF。

---

## 🏗️ リファクタリング詳細

### 旧構造の問題
case1 `baseline_v1` は `strategy.py` 702行に `plan_moves()` 内へすべてのロジックが集中していた。Mission 追加や movement 変更が 1 関数を巻き込み、回帰テスト・A/B 検証が困難。

### 新構造 (case2 baseline_v2)

```
pipeline/case2/baseline/
├── agent.py              (96行)   # 観測 → WorldModel → plan_moves の薄い wrapper
├── strategy.py           (168行)  # plan_moves orchestrator: collect_missions → dispatch → emit_movements
├── strategy_helpers.py   (396行)  # build_modes, preferred_send, target_value 等
├── opponent_model.py     (405行)  # OM v1 + v2 予測ロジック
├── lookahead.py          (143行)  # 1-ply self-play (default OFF)
├── core/
│   ├── config.py         # 全定数 (140+個の tuning param)
│   ├── world_model.py    (707行)  # WorldModel, simulate_planet_timeline, defense buffer
│   ├── physics.py, geometry.py, types.py
├── missions/
│   ├── __init__.py       # collect_missions() registry
│   ├── capture.py        # 攻撃・中立占領 + snipe
│   ├── swarm.py          # multi-source swarm
│   ├── reinforcement.py  # 自惑星強化
│   ├── crash_exploit.py  # 4P 限定: 敵同士の衝突を exploit
│   └── harass.py         # NEW: enemy production steal
└── movements/
    ├── followup.py       # 余剰艦の追加投入
    ├── evacuation.py     # 失陥予定惑星からの避難
    └── rear_guard.py     # 後方惑星の防衛配送
```

`plan_moves(world)` は以下の固定パイプラインに整理:
1. `build_modes(world)` — early/opening/late 等のモードフラグ
2. `collect_missions(world, planned, modes, ...)` — 5 builder が Mission を返す
3. Mission をスコア順 sort → 種類別に fleet を割り付け
4. `emit_followup_moves` / `emit_evacuation_moves` / `emit_rear_guard_moves`
5. dedup → final_moves

### 等価性検証

- `tests/pipeline/case2/test_baseline_agent.py` — seed 0 turn 10 の snapshot で action が case1 と完全一致
- `tests/pipeline/case2/test_world_model.py` — timeline, keep_needed, doomed 判定
- `tests/pipeline/case2/test_opponent_model.py` — 205行、launch detection / preference / prediction を網羅
- 全 17 test pass

### Kaggle 提出互換性

`pipeline/case<N>/main.py` は 20 行薄い wrapper、中身は `baseline/` 配下を相対 import のみで構成。`sys.path.insert(0, str(Path.cwd()))` で Kaggle ランタイム / ローカルどちらからも解決可能。

---

## 🔬 Phase 3: 機能追加と既存 flag の ablation

### 3-a. Harass mission の追加

**動機**: 既存 mission (capture/snipe/swarm/reinforce/crash_exploit) は「自分が取得する」or「自分を守る」行動のみ。敵 production を直接妨害する選択肢がなかった。

**設計**:
- 対象: `production >= HARASS_MIN_TARGET_PRODUCTION (=2)` の enemy planet
- 最小艦数で一時占領 → `HARASS_PRODUCTION_STEAL_TURNS (=5)` 程度 production を奪う
- スコア: `production * STEAL_TURNS * HARASS_VALUE_MULT / (need + turns * COST_TURN_WEIGHT + 1.0)`
- 出発元は `HARASS_MIN_SRC_RESERVE (=10)` 以上の艦を持つ my_planet のみ
- 到達 `HARASS_MAX_TRAVEL_TURNS (=20)` ターン以内に限定

**結果**: 100戦 seed 0 で 50-51% (ベースライン 47% から +3〜4pp、非有意だが正方向)

### 3-b. SAFE_INTERCEPT_HALF_STEP

`search_safe_intercept` の ETA 候補を 1 ターン刻み → 0.5 ターン刻みに拡張。分数ターンの射角補間で「軌道惑星の微妙なタイミング」を拾える。Plan.md で提案済みだった機能を有効化するだけ。

**結果**: 100戦 49% (+2pp、非有意)

### 3-c. Harass + HALF_STEP 同時 ON (最終採用構成)

- 100戦 seed 0: **57%**
- 100戦 seed 100: **55%**
- 300戦 seed 0: **50.7%**

seed 0 単発では +10pp という大きな改善に見えたが、**seed を変えると 55% に縮小**し、**300戦に拡大すると 50.7%** まで回帰。過去に OM v1 で観察された「seed variance に負ける」パターンが再現。

### 3-d/e/f. 既存 disabled flag の ablation

`config.py` に disabled 状態で存在していた 3 つの flag を個別に ON/OFF:

| flag | 100戦勝率 | 評価 |
|---|---|---|
| `DYNAMIC_PROACTIVE_HORIZON_ENABLED` | 47% | net-zero |
| `COMET_NPV_ENABLED` | 45% | **-6pp 害** |
| `FINISHING_TIE_GUARD` | 46% | **-11pp 害** |

特に `FINISHING_TIE_GUARD` は終盤引き分け回避ロジックのつもりが、攻撃閾値を押し上げて取り損ねを増やしていた可能性が高い。**全て OFF 維持**。

---

## 🤖 Phase 4: 構造的改修の試み

「100戦 seed variance ±8pp を破るには構造的改修が必要」という学びを元に、pointwise tuning ではなく情報量を増やす改修を 2 本試した。

### 4-a. OM v2 (phase-gated prediction)

**動機**: OM v1 は既に実装済みだが効果なし (memory 記録)。問題は「少ない launch 観測でも予測を出して score にノイズを足していた」可能性。v2 で改善:

- `OM_PREDICT_APPLY_AFTER_STEP (=30)` 未満は予測を返さない (序盤の false prediction 抑制)
- 各 enemy planet に `OM_PREDICT_MIN_LAUNCHES (=2)` 件以上の launch 観測を要求
- `ships_predicted = rate * my_share * OM_PREDICTION_WEIGHT (=0.5)` (uncertainty 反映)
- agent.py で `OM_V2_ENABLED` flag により v1/v2 を切替

**結果**:
- 100戦 seed 0: 53% (+6pp に見える)
- 100戦 seed 100: 48% (-2pp)
- 300戦 seed 0: **50.7%** (+1.4pp 非有意)

Harass+HALF_STEP と同一の着地点 (50.7%)。Phase-gate を追加しても prediction を score に加算する構造そのものが seed variance に埋もれる。

**判断**: 害はないので実装温存、default OFF。

### 4-b. Lookahead 1-ply self-play (初回)

**動機**: 「prediction を score に加算」ではなく、敵視点で `plan_moves` を実際に実行し、敵の最適反撃を `predicted_arrivals` へ注入することで WorldModel の `base_timeline` / `_compute_defense_buffers` に "相手の次の手" を反映させる。

**実装** (`pipeline/case2/baseline/lookahead.py`):

```python
def predict_enemy_fleets(world) -> dict[int, list[tuple[int, int, int]]]:
    # 1. 最強 enemy を特定
    enemy_id = _strongest_enemy(world)
    # 2. 同じ planets/fleets で player=enemy_id の WorldModel を構築
    enemy_world = WorldModel(player=enemy_id, ...)
    # 3. enemy_world に対して plan_moves() を呼ぶ (depth guard で無限再帰防止)
    enemy_moves = plan_moves(enemy_world)
    # 4. 敵が発射した fleet を target 推定して predicted_arrivals に変換
    #    ships_predicted = ships * LOOKAHEAD_PREDICTION_WEIGHT
```

`agent.py`:

```python
if LOOKAHEAD_ENABLED:
    probe = WorldModel(...)  # 1回目: 観測から直接作る
    lookahead_predictions = predict_enemy_fleets(probe)
    # 2回目: 敵の予測 fleet を predicted_arrivals に足して再構築
predicted_arrivals.update(lookahead_predictions)
return WorldModel(..., predicted_arrivals=predicted_arrivals)
```

**パラメータ**: `LOOKAHEAD_APPLY_AFTER_STEP=20`, `LOOKAHEAD_PREDICTION_WEIGHT=0.6`

**結果**:
- 100戦 seed 0: **42%** (**-8pp 害**)
- 100戦 seed 100: 51% (ほぼイーブン)
- 平均: **-3.5pp**
- turn_p95: 0.04s → 0.19s (**4倍重い**)

**原因分析**:
- 敵視点の `plan_moves` は我々が既に発射した fleet を arrivals として前提に最適反撃を組むため、**過剰予測** が混入
- weight=0.6 で予測艦数が `base_timeline` の `keep_needed` を押し上げ、**防御側の buffer が過剰化**
- 結果として `available[planet.id]` が減り、攻撃 mission に回す艦が不足

### 4-c. Lookahead gated (after_step=80, weight=0.3)

weight を下げ、序盤 80 ターンは lookahead OFF (greedy) にして過剰防御を抑える構成を試行。

**結果**:
- 100戦 seed 0: 50% (差無し)
- turn_p95: 0.19s (重さは変わらず)

**判断**: 軽微に改善したが有意差なし、かつ turn_p95 4倍の代償あり。**実装温存、default OFF**。

---

## 📊 seed variance の問題: 繰り返し確認されたパターン

`baseline_v2 vs baseline_v1` で 100戦 → 300戦拡大した際の勝率推移:

| 施策 | 100戦 seed0 | 100戦 seed100 | 300戦 seed0 | 100戦平均 | 300戦 |
|---|---|---|---|---|---|
| Harass + HALF_STEP | 57% | 55% | 50.7% | 56% | 50.7% |
| OM v2 | 53% | 48% | 50.7% | 50.5% | 50.7% |
| Lookahead (w=0.6) | 42% | 51% | — | 46.5% | — |
| Lookahead gated (w=0.3) | 50% | — | — | — | — |

**観察**:
1. 100戦単発結果は **±8pp のブレ** がある (Harass+HALF で seed 0 と seed 100 で 2pp 差、Lookahead で 9pp 差)
2. 勝率 50-55% に見えた改善は 300戦に拡大すると 50-51% へ収斂
3. 害 (-6pp 以上) は 100戦 seed 1 本でも検出可能 (効果サイズが variance を超える)

**運用ルール**:
1. 100戦の単発結果では **施策効果を判定しない** (seed 0 のみの improvement は seed variance 由来の可能性大)
2. 害の検出は 100戦 1 本で可 (有意差が出やすい)
3. 正方向の効果は **300戦以上、できれば別 seed 200戦を追加した 500戦** で追認
4. Harass+HALF_STEP は 300戦で害なし〜軽微 positive、採用して差し支えない

---

## 🧭 次ステップの方向性

### 現時点の把握

- case2 baseline_v2 は case1 baseline_v1 (Kaggle LB 897) に対し **300戦で 50.7% (+1.4pp 非有意)**
- Kaggle 提出実績: 2026-04-18 時点の旧 case2 版で **LB 784.2** (case1 v4 の 865.3 を下回る)
- 本レポートの構成 (Harass + HALF_STEP) は未提出のため、Kaggle LB 実測が必要

### 打ち止め判断

**pointwise な param tuning と prediction bonus 系はほぼ打ち止め**。以下の 2 本すら 300戦で +1.4pp に収束し有意差を得られなかった:
- OM v2 (phase-gated prediction + weight multiplier)
- Lookahead 1-ply (self-play による敵反撃予測)

### 残る候補

1. **RL 路線** (case3 新設, PPO/DQN で policy network 学習)
   - 期待効果: score の質的変化で seed variance を突破
   - コスト: 学習インフラ・報酬設計・自己対戦バッファなど数日規模
2. **MCTS / 探索ベース** (depth-limited game tree, random playout)
   - 期待効果: Lookahead が 1-ply で失敗したことから、depth を増やすよりは UCB/rollout で評価関数を変える方が筋が良い
   - コスト: overage time 内に収める工夫 (tight turn で破綻しやすい)
3. **AutoML / param grid search** (optuna 等で全 flag × 連続値の探索)
   - 期待効果: seed variance を突破できる可能性は低いが、個々の tuning で +0.5〜1pp 取れる可能性
   - コスト: 1 試行 20-60 分 × 数千試行 = 計算資源が課題
4. **現状維持 + Kaggle 提出 1 回** (skill rating を測って LB 順位を確認)
   - 期待効果: 情報量最大
   - コスト: 提出クォータ 1 消費

### 推奨

**提出 1 回** → LB 実測で case2 baseline_v2 の絶対位置を確認 → その結果次第で RL または AutoML を選択、が ROI 最大。
提出はクォータ消費 + 取り返しがつかない行為のため、**実施には user 承認が必要**。

---

## 🗃️ 実装成果物

### 新規ファイル
- `pipeline/case2/` 全体 (17 モジュール, 4900+ 行)
- `pipeline/case2/baseline/missions/harass.py` (107行)
- `pipeline/case2/baseline/lookahead.py` (143行, default OFF)
- `pipeline/case2/evaluation/snapshot_update.py` (開発用)
- `tests/pipeline/case2/` (3 ファイル, 17 test)

### 変更ファイル
- `src/env/agents.py` — AGENT_REGISTRY に `baseline_v2` 追加
- `pyproject.toml` — ruff 除外パス追加

### コミット履歴 (feature/optimize-case1)

```
a9074a5 :sparkles: 1-ply lookahead (self-play 反撃予測) を追加 (default OFF)
8dc3c8d :sparkles: OM v2 (phase-gated prediction) を追加 (default OFF)
1bee9d3 :sparkles: Harass mission と SAFE_INTERCEPT_HALF_STEP を有効化
8b30e24 :sparkles: case2 (baseline_v2) を追加 - mission/movement 分離構造で実装
```

### config.py の最終 flag 状態

| flag | default | 根拠 |
|---|---|---|
| `HARASS_ENABLED` | `True` | +3〜4pp (100戦), 害なし (300戦) |
| `SAFE_INTERCEPT_HALF_STEP` | `True` | +2pp (100戦), 害なし |
| `OPPONENT_MODEL_ENABLED` | `False` | OM v1/v2 とも 300戦で有意差なし |
| `OM_V2_ENABLED` | `False` | 300戦 +1.4pp, 実装温存 |
| `LOOKAHEAD_ENABLED` | `False` | -3.5pp 害 or 差無し + turn_p95 4倍 |
| `COMET_NPV_ENABLED` | `False` | -6pp 害 |
| `FINISHING_TIE_GUARD` | `False` | -11pp 害 |
| `DYNAMIC_PROACTIVE_HORIZON_ENABLED` | `False` | net-zero |
| `CRASH_EXPLOIT_ENABLED` | `True` | 4P 限定、既存のまま |
| `THREE_SOURCE_SWARM_ENABLED` | `True` | 既存のまま |
| `REINFORCE_ENABLED` | `True` | 既存のまま |

---

## 🔑 主要な学び (memory に記録済み)

1. **prediction bonus を score に加算する施策は seed variance に埋没する** — OM v1, OM v2, Lookahead 1-ply が全て +1-4pp の非有意か害に終わった
2. **100戦単発結果で +5〜10pp に見える改善は seed variance の範囲** — 300戦で 50-51% へ収束
3. **害 (-6pp 以上) は 100戦 1 seed で検出可能**
4. **構造的改修 (OM 再設計, lookahead) でも seed variance を破れない** — 次のブレイクスルーには評価関数そのものを置き換える改修 (RL, learned value function) が必要
5. **Lookahead の "敵視点 plan_moves を回す" 方式は敵の最適反撃を過剰予測** — 自分の発射を前提に防御を組むため weight を下げても過保守化しやすい
6. **turn_p95 の悪化は LB 提出前に見るべき** — Lookahead は 0.04s→0.19s で 4倍重くなった、actTimeout=1s を逼迫する

---

## 📚 参照

- 提案 plan: `docs/experiment/rulebase/20260418_case1_optimize/plan.md`
- 評価基盤: `docs/competition/20260418_evaluation.md`
- コンペ仕様: `docs/competition/abstract.md`
- Memory: `memory/project_case2_ablation.md`, `memory/project_om_finding.md`
