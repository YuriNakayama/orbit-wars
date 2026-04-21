# case5 (LB 1224 rulebase) — Codebase Research

対象: Kaggle notebook [`romantamrazov/orbit-star-wars-lb-max-1224`](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224) のロジックを `pipeline/rulebase/case5` として追加する。

## 1. Notebook (LB 1224) 構造分析

Kaggle CLI で取得済: `/tmp/lb1224/orbit-star-wars-lb-max-1224.ipynb` (単一セル、2,047 行、86 KB)。

### セクション構成（行番号）

| # | 範囲 | セクション | 主要内容 |
|---|------|-----------|---------|
| 1 | 7–177 | `Configuration` | 120+ のチューニング定数 (phase境界、攻撃cost、value multiplier、margin、defense/reinforce/recapture/rear/swarm/proactive defense/crash exploit/FFA/deadline 等) |
| 2 | 179–205 | `Shared Types` | `Planet` / `Fleet` namedtuple, `ShotOption` frozen dataclass, `Mission` dataclass |
| 3 | 208–385 | `Physics` | `dist`, `orbital_radius`, `is_static_planet`, `fleet_speed`, `segment_hits_sun`, `launch_point`, `actual_path_geometry`, `safe_angle_and_distance`, `predict_planet_position`, `predict_comet_position`, `comet_remaining_life`, `estimate_arrival`, `predict_target_position`, `target_can_move`, `search_safe_intercept`, `aim_with_prediction` |
| 4 | 387–589 | `World Model 準備ユーティリティ` | `fleet_target_planet`, `build_arrival_ledger`, `resolve_arrival_event`, `normalize_arrivals`, `simulate_planet_timeline`, `state_at_timeline`, `count_players`, `nearest_distance_to_set`, `indirect_features`, `detect_exposed_enemy_planets`, `detect_enemy_fights_at_neutrals` |
| 5 | 592–867 | `WorldModel クラス` | 276 行。世界状態の集約＋キャッシュ（owner_strength, base_timeline, keep_needed_map, fall_turn_map, holds_full_map, indirect_feature_map, exposed_planet_ids, blood_in_water_owners, shot_cache, reaction_cache 等） |
| 6 | 868–1504 | `Strategy helpers` | `planet_distance`, `nearest_sources_to_target`, `min_legal_reaction_time`, `policy_reaction_times`, `stacked_enemy_proactive_keep`, `swarm_eta_tolerance`, `detect_enemy_crashes`, `build_policy_state`, `build_modes`, `is_safe_neutral`, `is_contested_neutral`, `opening_filter`, `target_value`, `reinforce_value`, `preferred_send`, `apply_score_modifiers`, `settle_plan`, `settle_reinforce_plan`, `build_snipe_mission`, `build_rescue_missions`, `build_recapture_missions`, `build_reinforce_missions`, `build_crash_exploit_missions` |
| 7 | 1506–1985 | `plan_moves` (480 行の大関数) | missions の統合、優先度ソート、defensive check、multi-source swarm、proactive defense、evacuation、rear-guard、followup 全てをインラインで実装 |
| 8 | 1987–2045 | `Agent Entry Point` | `_read`, `_detect_game_id`, `build_world`, `agent(obs, config=None)` — `time.perf_counter()` で `SOFT_ACT_DEADLINE` ベースの soft budget を作って `plan_moves(world, deadline=deadline)` に渡す |

### LB 1224 notebook の独自機能（case4 に無い機能）

コードベース diff で確認した notebook 固有の機能:

1. **deadline ベースの時間制御**: `plan_moves(world, deadline)` に `actTimeout` から計算した deadline を渡し、重いフェーズ (`HEAVY_PHASE_MIN_TIME=0.16`, `OPTIONAL_PHASE_MIN_TIME=0.08`) をスキップする仕組み。case4 にはこの機構が無い。
2. **`detect_enemy_crashes` + crash-exploit mission の完成形**: 敵の fleet が太陽に落ちる場合を予測して exploit。case4 にも `crash_exploit.py` は存在するが、notebook 側はより洗練 (`CRASH_EXPLOIT_ETA_WINDOW=2`, `POST_CRASH_DELAY=1`, スコア multiplier まで導入)。
3. **`build_recapture_missions`**: 最近失った惑星を取り戻す専用ミッション (`RECAPTURE_LOOKAHEAD_TURNS=10`, `RECAPTURE_VALUE_MULT=0.88`, `RECAPTURE_FRONTIER_MULT=1.08`)。case4 には無い。
4. **`build_rescue_missions`**: 敵の攻撃で落ちる自軍惑星を救援する専用ミッション。case4 は defense が `plan_moves` 内にインラインで散在。
5. **`detect_exposed_enemy_planets`**: 艦隊を出して手薄になった敵惑星を検出して価値を 1.85x に嵩上げ (`EXPOSED_PLANET_VALUE_MULT`)。case4 には無い概念。
6. **`blood_in_water_owners` (FFA 殲滅戦)**: `FFA_ELIMINATION_SHIPS=55` 以下の弱った敵を検出、`BLOOD_IN_WATER_VALUE_MULT=1.45` + `ELIMINATION_BONUS=32.0` + `ELIMINATION_PROD_BONUS=18.0`。case4 にも weakest_enemy 概念はあるが、scoring がより精緻。
7. **`detect_enemy_fights_at_neutrals`**: 敵同士が同じ中立を狙っている場合 `LET_THEM_FIGHT_PENALTY=0.72` で自軍参戦を抑制。case4 には無い。
8. **multi-source swarm の 3 段階化**: notebook は `MULTI_SOURCE_TOP_K=5`, `THREE_SOURCE_SWARM_ENABLED=True`, `THREE_SOURCE_MIN_TARGET_SHIPS=18` で 2/3 源スワームを使い分け。case4 の `swarm.py` は 2源固定に近い。
9. **`stacked_enemy_proactive_keep` + proactive defense**: 複数敵が同一惑星へ同時着弾する stack window (`MULTI_ENEMY_STACK_WINDOW=3`) を検出、`PROACTIVE_DEFENSE_HORIZON=12` で予防防御 reserve を計算。case4 には無い。
10. **`indirect_features` / `apply_score_modifiers`**: 周辺惑星の owner 分布 (friendly/neutral/enemy) を `INDIRECT_VALUE_SCALE=0.15` で加点する。case4 には無い indirect value 概念。
11. **`build_modes` + mode 駆動の value 計算**: ahead/behind/finishing/total_war/is_four_player など複数の mode フラグでスコアを二段階補正。case4 の `strategy_helpers.build_modes` と比べて補正パラメタが倍以上。
12. **opponent_model (OM) 無し**: notebook は opponent 推定を持たない。case4 は `opponent_model.py` + `lookahead.py` で予測。

### notebook に**無い** case4 独自機能

- `opponent_model.py` (405 行): 敵の launch 履歴から preferences を推定、`predict_future_arrivals_v2` で将来到達予測。
- `lookahead.py` (141 行): `predict_enemy_fleets` で敵の次手を予測して `predicted_arrivals` に注入。
- `harass.py`: ハラスメント専用ミッション (notebook には無い)。
- `evaluation/` ディレクトリ: ablation / compare_v2 / snapshot_update スクリプト。
- `configs/baseline.yaml`: YAML ベースの ablation 設定。

### 規模比較

| 指標 | LB1224 notebook | case4 (baseline_v4) |
|------|----------------|--------------------|
| 総行数 | 2,047 | 3,676 (分割後) |
| 関数/クラス数 | 60 | ~80 |
| `plan_moves` の行数 | 480 (単一関数) | 167 (薄いオーケストレーション) + mission モジュール群 |
| Mission 種類 | capture / snipe / rescue / recapture / reinforce / crash_exploit / swarm (implicit) | capture / snipe / reinforcement / swarm / harass / crash_exploit |
| deadline 制御 | あり | 無し |
| opponent model | 無し | あり (OFF 推奨 — memory 参照) |

## 2. 既存 pipeline のパターン (Kaggle submission 規約)

`.claude/rules/pipeline.md` の規約:

- `pipeline/<category>/case<N>/main.py` が必ず case ディレクトリ直下に存在し、トップレベルで `agent(obs)` を公開。
- `main.py` は 20 行程度の wrapper: `sys.path.insert(0, str(Path.cwd()))` + `from baseline.agent import agent`。
- サブパッケージ内部は**すべて相対 import** で統一 (`from ..core.types import Fleet`)。
- `__file__` を sys.path 注入に使ってはいけない（Kaggle で Validation failed）。必ず `Path.cwd()`。
- 実装本体は `pipeline/<category>/case<N>/<package>/` のサブパッケージとして階層化。case4 では `baseline/` 配下に `core/`, `missions/`, `movements/` と分割。
- `pipeline/.submitignore` で `eda/`, `evaluation/`, `configs/`, `notebook/`, `training/` を除外。

### case4 の分割方針（参考にすべき）

```
pipeline/rulebase/case4/
├── main.py                              # 20行 wrapper
├── baseline/
│   ├── __init__.py                      # from .agent import agent
│   ├── agent.py                         # 124行: build_world + agent + OM update
│   ├── strategy.py                      # 167行: plan_moves（薄いオーケストレーション）
│   ├── strategy_helpers.py              # 403行: target_value, build_modes, settle_plan
│   ├── lookahead.py                     # 141行
│   ├── opponent_model.py                # 405行
│   ├── core/
│   │   ├── config.py                    # 204行: 定数
│   │   ├── geometry.py                  # 75行: 幾何プリミティブ
│   │   ├── physics.py                   # 279行: orbit prediction
│   │   ├── types.py                     # 50行: Planet/Fleet/ShotOption/Mission
│   │   └── world_model.py               # 707行: WorldModel class
│   ├── missions/                        # 各ミッションを 1 ファイル 80-160 行に分割
│   │   ├── __init__.py                  # collect_missions() 集約
│   │   ├── capture.py                   # 157行
│   │   ├── crash_exploit.py             # 89行
│   │   ├── harass.py                    # 109行
│   │   ├── reinforcement.py             # 108行
│   │   ├── snipe.py                     # 97行
│   │   └── swarm.py                     # 157行
│   └── movements/                       # 着弾後/退避/後方警備 の個別ロジック
│       ├── evacuation.py                # 109行
│       ├── followup.py                  # 121行
│       └── rear_guard.py                # 111行
├── configs/baseline.yaml
└── evaluation/
    ├── ablation.py
    ├── compare_v2.py
    └── snapshot_update.py
```

### `src/dataset/selfplay/agents.py` 登録

```python
AGENT_REGISTRY: dict[str, str] = {
    "baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent",
    ...
    "baseline_v4": "pipeline.rulebase.case4.baseline.agent:agent",
    # ここに "baseline_v5" 追加想定
    "case0": "pipeline.rulebase.case0.main:agent",
    ...
}
```

## 3. テスト構成

`tests/pipeline/rulebase/case4/`:

- `test_baseline_agent.py`: `env.run([agent, "random"])` で 1 局走らせ、`status in {"DONE","ACTIVE"}` を確認。
- `test_fleet_consolidation.py`: movements/followup の検証。
- `test_opponent_model.py`: OM の挙動。
- `test_world_model.py`: WorldModel の timeline 計算を直接検証。

case5 も同水準を想定 (agent smoke + world_model core + 各 mission の単体テスト)。

## 4. 技術的制約

1. **Kaggle actTimeout=1s / step**: notebook は `deadline = start + min(SOFT_ACT_DEADLINE, act_timeout*0.82)` で計算。case5 でもこの機構を維持する必要あり (特に `plan_moves` が重いため)。
2. **`__file__` 使用禁止**: `main.py` では `Path.cwd()` のみ使用 (case4 と同パターン)。
3. **単一ファイルを分割する際の識別性**: notebook は全関数がトップレベル同一スコープにあるため、分割時は相互依存に注意（`WorldModel` は helper 関数を多数呼ぶ）。case4 では `core/world_model.py` に `build_arrival_ledger` 等も集約して循環を回避している。
4. **opponent_model の判断**: memory (`project_om_finding.md`) によると OM 施策は 100 戦評価で有意改善なし、デフォルト OFF。LB1224 notebook は OM 非搭載でも 1224 に到達しているため、**case5 では OM/lookahead を持ち込まない** 選択が妥当。
5. **case4 との比較評価**: case4 は 300 戦で baseline_v2 比 +10pp (`project_case4_phase_results.md`)。case5 は notebook の LB 実績 (MAX 1224) ベースのため、case4 との自己対戦で定量比較が必要。

## 5. 主要な再利用候補と Gap

### 再利用可能（case4 → case5 でほぼそのまま流用可）

- `core/geometry.py` (75行): `dist`, 基本幾何。notebook と完全同等。
- `core/physics.py` (279行): `orbital_radius`, `fleet_speed`, `segment_hits_sun`, `predict_planet_position`, `predict_comet_position`, `aim_with_prediction` 等。notebook 側の関数はほぼ一致。
- `core/types.py`: `Planet`, `Fleet` namedtuple。`ShotOption`, `Mission` は notebook 版 (field 差分あり) を採用。
- `main.py` のテンプレート: 20 行 wrapper をそのまま流用。

### 流用不可 / 新規実装必要

- `core/config.py`: 定数値が大幅に異なる (80+ の新規定数; `SIM_HORIZON=110`, `ROUTE_SEARCH_HORIZON=60`, `SOFT_ACT_DEADLINE=0.82` 等)。
- `core/world_model.py`: `blood_in_water_owners`, `enemy_fights`, `exposed_planet_ids`, `indirect_feature_map` 等 case4 に無い属性を追加必要。`predicted_arrivals` / `opponent_threat_score` は case5 では削除。
- `strategy_helpers.py`: `build_policy_state`, `stacked_enemy_proactive_keep`, `swarm_eta_tolerance`, `detect_enemy_crashes`, `opening_filter`, `apply_score_modifiers`, `settle_reinforce_plan`, `reinforce_value` が新規。
- `missions/`: `rescue.py`, `recapture.py` 新規。`capture.py` / `snipe.py` / `swarm.py` / `crash_exploit.py` / `reinforcement.py` は notebook のロジックに合わせて再実装。`harass.py` は notebook に無いため削除。
- `strategy.py` (plan_moves): notebook の 480 行インライン版を、case4 式に薄い orchestration + missions 呼び出しに分解する必要。ここが最大の作業。
- `agent.py`: `_game_id` 検出、`time.perf_counter()` + `actTimeout` ベースの deadline 制御を導入（case4 にはない）。OM 連携は削除。

## 6. Key Findings Summary

1. **notebook は case4 と同系ファミリーだが進化方向が異なる**: notebook は OM を捨て、その代わりに deadline 制御・rescue/recapture/crash-exploit/FFA 殲滅などのミッション多様化で LB 1224 を達成。
2. **case4 の分割パターンがそのまま適用可能**: `baseline/core/`, `baseline/missions/`, `baseline/movements/`, `baseline/strategy.py` の構造を踏襲すれば、notebook の 2,047 行を 200-400 行 × 10-12 ファイルに分解できる (backend.md の「200-400 行/ファイル」規範に適合)。
3. **deadline 制御の移植が必須**: notebook の strength は "最も重いフェーズ (heavy route search) を時間切れで skip する" 点にあり、これを削ると品質が落ちる可能性が高い。`core/config.py` に `SOFT_ACT_DEADLINE` / `HEAVY_PHASE_MIN_TIME` / `HEAVY_ROUTE_PLANET_LIMIT` を定数として持ち込み、`strategy.py` の各ミッション呼び出し前に残り時間をチェックする。
4. **移植スコープの選択肢が複数あり**: (A) 全機能を忠実移植、(B) core + capture/snipe/swarm のみ先に移植しイテレーション、(C) notebook 定数のみ流用して case4 を調整。ユーザーに選んでもらう必要あり。
5. **ライセンス/帰属**: notebook は Kaggle 公開コード。case4 と同様に冒頭コメントで "Adapted from ..." と出典明記する必要 (Apache License の可能性あり、要確認)。
