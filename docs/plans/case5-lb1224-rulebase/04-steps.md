# case5 (LB 1224 rulebase) — 実装ステップ

## 戦略

- **Backend-first / Bottom-up**: 依存方向に沿って `core/` → `world_model.py` → `missions/` → `strategy.py` → `agent.py` → tests の順に積み上げる
- **同時に notebook L7-2045 を等価コピー後、規範準拠リファクタ** という 2 段階アプローチ
- 1 Step = 1 Pull Request のサイズ感 (200-400 行/PR 目安)
- 各 Step 完了時に `dev/test-backend` が緑であることを必須

## 依存関係グラフ

```
Step 01 (case dir + main.py + LICENSE)
    └─ Step 02 (core/types, core/geometry, core/config)
        └─ Step 03 (core/physics)
            └─ Step 04 (core/timing) ─┐
                                       │
            └─ Step 05 (core/world_model 基本部) ─┐
                  └─ Step 06 (world_model 拡張: exposed/blood/fights/indirect/stacked)
                                                   │
        └─ Step 07 (strategy_helpers: build_modes/target_value/settle_plan)
            └─ Step 08 (missions/capture, snipe, reinforcement) ─┐
            └─ Step 09 (missions/swarm: 2源/3源)                  │
            └─ Step 10 (missions/rescue, recapture)               │
            └─ Step 11 (missions/crash_exploit)                   │
            └─ Step 12 (missions/__init__: collect_missions)      │
                                                                  │
        └─ Step 13 (movements/evacuation, rear_guard, followup)   │
            └─ Step 14 (movements/proactive_defense)              │
                                                                  │
        └─ Step 15 (strategy.plan_moves: collect → resolve → apply)
            └─ Step 16 (agent.py + AGENT_REGISTRY 登録)
                └─ Step 17 (tests: smoke + snapshot + 単体)
                    └─ Step 18 (configs/baseline.yaml + evaluation/)
                        └─ Step 19 (notebook 等価性 / 自己対戦検証)
                            └─ Step 20 (Kaggle dry-run + 提出準備)
```

並列化可能: **Step 08, 09, 10, 11** は Step 07 完了後に並行作業可。**Step 13, 14** は Step 06 完了後に並行可。

---

## Step 01: case ディレクトリ初期化 + LICENSE

**Target**: cross-cutting
**Dependencies**: None

### Overview
case5 のディレクトリ骨格と Apache 2.0 ライセンスファイルを配置。

### Work Items
- [ ] `pipeline/rulebase/case5/main.py` (20 行 wrapper、case4 と同テンプレ、`Path.cwd()` 注入)
- [ ] `pipeline/rulebase/case5/baseline/__init__.py` (出典コメント + `from .agent import agent`)
- [ ] `pipeline/rulebase/case5/baseline/LICENSE` (Apache 2.0 全文)
- [ ] `pipeline/rulebase/case5/baseline/core/__init__.py`
- [ ] `pipeline/rulebase/case5/baseline/missions/__init__.py`
- [ ] `pipeline/rulebase/case5/baseline/movements/__init__.py`
- [ ] `pipeline/rulebase/case5/configs/__init__.py` (空)
- [ ] `pipeline/rulebase/case5/evaluation/__init__.py` (空)
- [ ] `tests/pipeline/rulebase/case5/__init__.py`

### Acceptance Criteria
- `uv run python -c "import pipeline.rulebase.case5.baseline"` が ImportError 無く通る (中身が空でも可)
- LICENSE ファイルに NOTICE で notebook 出典明記

---

## Step 02: core/types, core/geometry, core/config

**Target**: backend
**Dependencies**: Step 01

### Overview
最も基礎的な型定義・幾何計算・全 120+ 定数を移植。

### Work Items
- [ ] `core/types.py`: `Planet`/`Fleet` namedtuple, `ShotOption`/`Mission` を `frozen=True, slots=True` の dataclass で定義 (`with_score()` ヘルパー含む)
- [ ] `core/geometry.py`: `dist`, 基本幾何 (notebook L208-220 相当)
- [ ] `core/config.py`: notebook L7-177 の全定数を移植。セクション (PHASE / ATTACK / VALUE / MARGIN / DEFENSE / SWARM / FFA / DEADLINE 等) ごとにコメント区切り

### Acceptance Criteria
- `mypy --strict pipeline/rulebase/case5/baseline/core/` がエラー 0
- 全定数は `Final` annotation 付き
- `Mission.with_score(0.5)` が新インスタンスを返す単体テスト追加

---

## Step 03: core/physics

**Target**: backend
**Dependencies**: Step 02

### Overview
軌道予測・コメット予測・狙い計算を移植。

### Work Items
- [ ] `core/physics.py`: notebook L208-385 から幾何プリミティブを除く全関数を移植
  - `orbital_radius`, `is_static_planet`, `fleet_speed`
  - `segment_hits_sun`, `launch_point`, `actual_path_geometry`, `safe_angle_and_distance`
  - `predict_planet_position`, `predict_comet_position`, `comet_remaining_life`
  - `estimate_arrival`, `predict_target_position`, `target_can_move`
  - `search_safe_intercept`, `aim_with_prediction`
- [ ] case4 の `core/physics.py` 既存関数とシグネチャを比較。差分があれば notebook 側を採用 (notebook 由来の正確な仕様を維持)

### Acceptance Criteria
- 各関数で 1-3 個の単体テスト (静止惑星 / 軌道惑星 / コメット のサンプル obs)
- ベクトル化されている関数は NumPy で実装
- 純粋関数 (副作用なし、引数だけで結果が決まる)

---

## Step 04: core/timing (★新規)

**Target**: backend
**Dependencies**: Step 02

### Overview
deadline 制御の中央集約モジュールを新設。

### Work Items
- [ ] `core/timing.py`:
  - `Deadline` frozen dataclass (`started_at`, `deadline_at`)
  - `from_config(config, *, soft_max, fraction)` クラスメソッド
  - `remaining(now=None)` インスタンスメソッド
  - `should_skip(min_time)` インスタンスメソッド
- [ ] `core/config.py` への定数追加: `SOFT_ACT_DEADLINE`, `SOFT_ACT_FRACTION`, `HEAVY_PHASE_MIN_TIME`, `OPTIONAL_PHASE_MIN_TIME`

### Acceptance Criteria
- `tests/.../test_timing.py` で monkeypatch 経由の fake clock テスト
  - `should_skip` が `remaining < min_time` で True を返す
  - `from_config` が `actTimeout=0.5` のとき `deadline_at = start + min(0.82, 0.5*0.82) = start + 0.41`
- monkeypatch 不要 / 副作用最小

---

## Step 05: core/world_model 基本部

**Target**: backend
**Dependencies**: Step 03, 04

### Overview
WorldModel の core フィールド (case4 互換部) と timeline 計算を移植。

### Work Items
- [ ] `core/world_model.py` Part 1:
  - `fleet_target_planet`, `build_arrival_ledger`, `resolve_arrival_event`, `normalize_arrivals`
  - `simulate_planet_timeline`, `state_at_timeline`, `count_players`
  - `WorldModel` クラス (constructor, `owner_strength`, `base_timeline`, `keep_needed_map`, `fall_turn_map`, `holds_full_map`, `shot_cache`, `reaction_cache`)
- [ ] `nearest_distance_to_set` ヘルパー

### Acceptance Criteria
- `test_world_model.py` で固定 obs に対し timeline 計算が想定値と一致 (case4 の test_world_model.py をベース)
- `simulate_planet_timeline` が arrival events を時系列ソートして処理することを単体テストで検証

---

## Step 06: world_model 拡張 (notebook 固有機能)

**Target**: backend
**Dependencies**: Step 05

### Overview
notebook 固有の検出ロジックを WorldModel に追加。

### Work Items
- [ ] `indirect_features` 関数 (周辺惑星 owner 分布)
- [ ] `detect_exposed_enemy_planets` → `WorldModel.exposed_planet_ids`
- [ ] `detect_enemy_fights_at_neutrals` → `WorldModel.enemy_fights_at_neutrals`
- [ ] blood-in-water 検出: `WorldModel.blood_in_water_owners` (FFA で `ships ≤ FFA_ELIMINATION_SHIPS` の player 集合)
- [ ] `stacked_enemy_proactive_keep` → `WorldModel.stacked_enemy_keep`

### Acceptance Criteria
- 各検出関数に単体テスト (対応する状況の固定 obs を構築)
- `exposed_planet_ids` が実際に手薄な敵惑星のみを含む
- `blood_in_water_owners` が FFA 限定で動作することを test_world_model.py で確認

---

## Step 07: strategy_helpers

**Target**: backend
**Dependencies**: Step 06

### Overview
mode 駆動のスコアリング・settle・mission 補正ロジック。

### Work Items
- [ ] `strategy_helpers.py`:
  - `planet_distance`, `nearest_sources_to_target`
  - `min_legal_reaction_time`, `policy_reaction_times`, `swarm_eta_tolerance`
  - `build_policy_state`, `build_modes` (`ModeFlags` frozen dataclass)
  - `is_safe_neutral`, `is_contested_neutral`, `opening_filter`
  - `target_value`, `reinforce_value`, `preferred_send`
  - `apply_score_modifiers` (indirect / exposed / blood-in-water / let-them-fight 全部適用)
  - `settle_plan` (採用 mission を world に反映)
  - `settle_reinforce_plan`

### Acceptance Criteria
- `test_strategy_helpers.py` で:
  - `build_modes` が ahead/behind/finishing/total_war の各シナリオを正しく検出
  - `apply_score_modifiers` が exposed_planet target の score を `EXPOSED_PLANET_VALUE_MULT` 倍にする
  - `settle_plan` で消費された ships が次の mission スコア計算に反映される

---

## Step 08: missions capture / snipe / reinforcement (並列可)

**Target**: backend
**Dependencies**: Step 07

### Overview
基本ミッションを移植。

### Work Items
- [ ] `missions/capture.py`: `build_capture_missions(world, *, deadline)` を notebook L1500-1700 から抽出 (plan_moves のインライン部から関数化)
- [ ] `missions/snipe.py`: `build_snipe_missions` を notebook `build_snipe_mission` から複数化
- [ ] `missions/reinforcement.py`: `build_reinforce_missions` 移植

### Acceptance Criteria
- 各ファイルに 80-160 行
- 1 mission = 1 builder 関数、純粋関数化 (WorldModel + Deadline 入力 → list[Mission] 出力)
- 各 mission に 1-2 個の単体テスト

---

## Step 09: missions/swarm (並列可)

**Target**: backend
**Dependencies**: Step 07

### Overview
multi-source swarm の 2源/3源切替ロジックを抽出。

### Work Items
- [ ] `missions/swarm.py`:
  - `build_swarm_missions(world, *, deadline)`
  - 2源パス (デフォルト)
  - 3源パス (`THREE_SOURCE_SWARM_ENABLED=True` かつ target ships ≥ `THREE_SOURCE_MIN_TARGET_SHIPS`)
  - `MULTI_SOURCE_TOP_K` で source 候補絞り込み

### Acceptance Criteria
- 単体テストで 2源/3源切替閾値の境界値を確認
- target_value が高い時のみ 3源スワームを試みる

---

## Step 10: missions/rescue + recapture (★新規, 並列可)

**Target**: backend
**Dependencies**: Step 07

### Overview
notebook 独自の防御/再奪還ミッションを抽出。

### Work Items
- [ ] `missions/rescue.py`: `build_rescue_missions` (落ちる自軍惑星の救援、`fall_turn_map` を参照)
- [ ] `missions/recapture.py`: `build_recapture_missions` (`RECAPTURE_LOOKAHEAD_TURNS`, `RECAPTURE_VALUE_MULT`, `RECAPTURE_FRONTIER_MULT`)

### Acceptance Criteria
- `test_missions_rescue.py`: 落下予測がある自軍惑星に対し rescue mission が生成される
- `test_missions_recapture.py`: 直近で失った惑星に対し recapture mission が生成され、frontier の場合 1.08x

---

## Step 11: missions/crash_exploit (並列可)

**Target**: backend
**Dependencies**: Step 07

### Overview
敵 fleet 太陽落下を予測した exploit mission。

### Work Items
- [ ] `missions/crash_exploit.py`:
  - `detect_enemy_crashes` (notebook L1100-1130)
  - `build_crash_exploit_missions` (`CRASH_EXPLOIT_ETA_WINDOW=2`, `POST_CRASH_DELAY=1`)

### Acceptance Criteria
- `test_missions_crash_exploit.py`: 敵 fleet が太陽に向かう状況の fixed obs で exploit mission が生成

---

## Step 12: missions/__init__.py (collect_missions)

**Target**: backend
**Dependencies**: Step 08-11

### Overview
全 mission builder を集約し、deadline スキップ判定を組み込む。

### Work Items
- [ ] `missions/__init__.py`:
  - `collect_missions(world, *, deadline)` (重いフェーズの should_skip 判定込み)
  - `apply_score_modifiers` を集約後に呼ぶ (strategy_helpers 経由)

### Acceptance Criteria
- 単体テストで deadline 切迫時に重いフェーズがスキップされることを確認 (monkeypatch で deadline 操作)

---

## Step 13: movements 既存系 (evacuation/rear_guard/followup, 並列可)

**Target**: backend
**Dependencies**: Step 06

### Overview
case4 の同等モジュールを参照しつつ、notebook 仕様に合わせて移植。

### Work Items
- [ ] `movements/evacuation.py`: 太陽落下回避退避 (notebook L1850-1900)
- [ ] `movements/rear_guard.py`: 後方警備 (notebook L1900-1950)
- [ ] `movements/followup.py`: 連続着弾 (notebook L1950-1985)

### Acceptance Criteria
- 各ファイル 100-130 行
- 単体テストでそれぞれ 1 シナリオ確認

---

## Step 14: movements/proactive_defense (★新規)

**Target**: backend
**Dependencies**: Step 06

### Overview
複数敵 stack window 検出による予防防御 reserve 計算。

### Work Items
- [ ] `movements/proactive_defense.py`:
  - `build_proactive_defense(world, *, deadline)` (`MULTI_ENEMY_STACK_WINDOW=3`, `PROACTIVE_DEFENSE_HORIZON=12`)
  - WorldModel.stacked_enemy_keep を消費

### Acceptance Criteria
- `test_movements_proactive.py`: 同一惑星に複数敵が stack window 内に着弾する fixed obs で reserve が増える

---

## Step 15: strategy.py (plan_moves)

**Target**: backend
**Dependencies**: Step 12, 13, 14

### Overview
notebook の 480 行 `plan_moves` を 3 段階に分解した薄いオーケストレーション。

### Work Items
- [ ] `strategy.py`:
  - `plan_moves(world, *, deadline)` ~30 行
  - `_collect → _resolve → _apply` の各内部関数 (~40 行)
  - 採用 mission の score sort と settle_plan ループ

### Acceptance Criteria
- ファイル合計 < 200 行
- 関数あたり 50 行以内
- 既存 case4 の strategy.py のパターンを参考に書く

---

## Step 16: agent.py + AGENT_REGISTRY

**Target**: cross-cutting
**Dependencies**: Step 15

### Overview
エージェントエントリポイントと selfplay レジストリ登録。

### Work Items
- [ ] `agent.py`:
  - `_read(obs)`, `_detect_game_id(obs)` (notebook L1987-2010)
  - `build_world(obs)` (NumPy 化された WorldModel 構築)
  - `agent(obs, config=None)` で `Deadline.from_config` → `plan_moves`
- [ ] `src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に `"baseline_v5": "pipeline.rulebase.case5.baseline.agent:agent"` 追加

### Acceptance Criteria
- `uv run python -c "from pipeline.rulebase.case5.baseline.agent import agent; print(agent)"` が通る
- `uv run python -c "from pipeline.rulebase.case5.main import agent; print(agent)"` も通る (sys.path 注入後)
- `agent({...}, {"actTimeout": 1.0})` が `list[list]` を返す

---

## Step 17: tests (smoke + snapshot + 主要単体)

**Target**: tests
**Dependencies**: Step 16

### Overview
case4 と同水準のテストスイート。

### Work Items
- [ ] `test_baseline_agent.py`:
  - `env.run([agent, agent])` 1v1 完走テスト (`status in {"DONE","INACTIVE"}`)
  - action shape テスト (synthetic obs)
  - snapshot test (固定 obs に対する action 一致)
- [ ] 各 mission/world_model/strategy_helpers/timing の単体テスト (Step 02-14 で逐次追加したものを集約)

### Acceptance Criteria
- `pytest tests/pipeline/rulebase/case5/` がすべて通る
- `pytest --cov=pipeline/rulebase/case5/baseline tests/pipeline/rulebase/case5/` でカバレッジ 80% 以上

---

## Step 18: configs + evaluation

**Target**: cross-cutting
**Dependencies**: Step 17

### Overview
ablation 設定と比較スクリプト。

### Work Items
- [ ] `configs/baseline.yaml`: 各 mission on/off + 主要定数 override (case4 の baseline.yaml を参考)
- [ ] `evaluation/ablation.py`: 設定別に N 戦自己対戦して勝率を出す
- [ ] `evaluation/compare_v2.py`: case4 vs case5 の対戦勝率
- [ ] `evaluation/snapshot_update.py`: snapshots 更新スクリプト
- [ ] `pipeline/.submitignore` の確認 (`evaluation/`, `configs/` は既存設定で除外される)

### Acceptance Criteria
- `uv run python -m pipeline.rulebase.case5.evaluation.snapshot_update` で snapshots 更新可
- `uv run python -m pipeline.rulebase.case5.evaluation.compare_v2 -n 10` でテスト走行可

---

## Step 19: notebook 等価性検証 / 自己対戦

**Target**: cross-cutting
**Dependencies**: Step 18

### Overview
リファクタによる挙動変化が無いことを定量確認。

### Work Items
- [ ] notebook を `/tmp/lb1224/` から実行可能にし、固定 obs に対する action を採取
- [ ] case5 の同 obs に対する action と比較 → 主要 obs 5-10 個で完全一致を確認 (deadline 微差は許容)
- [ ] case4 vs case5 を 100 戦自己対戦、勝率を記録 (≥ 45% を目標)
- [ ] memory に結果を記録 (`project_case5_validation.md`)

### Acceptance Criteria
- snapshot test 5-10 個が等価
- case4 vs case5 100 戦勝率 ≥ 45%

---

## Step 20: Kaggle dry-run + 提出準備

**Target**: cross-cutting
**Dependencies**: Step 19

### Overview
Kaggle 提出に向けた最終検証。本番提出はユーザー承認を得てから。

### Work Items
- [ ] `dev/test-backend` (format/lint/type/pytest) 全部緑
- [ ] `uv run python -m submit submit rulebase/case5 --dry-run -m "case5 baseline_v5 dry-run"` を **`--skip-validation` 併用** または case5 ディレクトリで実行
- [ ] 提出メッセージ案を作成: "case5: port of LB1224 notebook (mission diversification + deadline control)"
- [ ] **本番提出はユーザー承認を取得してから実行** (CLAUDE.md ルール準拠)

### Acceptance Criteria
- dry-run でアーカイブが作成され、必要ファイルのみ含まれる
- LICENSE / 出典コメントが含まれる
- ユーザー承認後、`SubmissionStatus.ERROR` 以外で受理される

---

## クロスカット注意点

| 観点 | 対応 |
|------|------|
| **import 経路** | サブパッケージ内部はすべて相対 import (`from ..core.types import Mission`)。`__file__` 使用禁止 |
| **ライセンス** | Step 01 で配置、Step 17 のテストで `LICENSE` 存在確認テストを追加してもよい |
| **case4 への影響** | 一切触らない。case5 は完全に独立 |
| **dev/format / dev/lint** | 各 Step 完了時に必ず実行 |
| **コミット粒度** | 1 Step = 1 commit を目安、ただし Step 02/03 など機能追加と同時に test も含めて 1 PR でまとめる |
