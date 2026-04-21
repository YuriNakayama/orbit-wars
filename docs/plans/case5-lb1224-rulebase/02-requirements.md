# case5 (LB 1224 rulebase) — 要件定義

## 1. 背景と目的

Kaggle [`romantamrazov/orbit-star-wars-lb-max-1224`](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224) は Public LB 1224 の実績を持つ Orbit Wars エージェント notebook (Apache 2.0)。
本プロジェクトの case4 (baseline_v4) は OM/lookahead 路線で baseline_v2 比 +10pp の改善を達成しているが、notebook は **OM 非搭載・mission 多様化・deadline 制御** という別路線で LB1224 に到達している。

**目的**: notebook と等価のロジックを `pipeline/rulebase/case5/` として実装し、`baseline_v5` として AGENT_REGISTRY に登録する。同時に notebook の単一巨大関数構造を case4 のサブパッケージ分割パターンに合わせて再編し、**可読性・テスト性を獲得**しながら notebook 同等の対戦性能を維持する。

## 2. ユーザーストーリー

| 優先度 | ストーリー | 受け入れ条件 |
|--------|----------|-------------|
| P0 | エージェント開発者として、case5 を `baseline_v5` で selfplay 評価したい | `uv run python -m dataset selfplay --agents baseline_v4 baseline_v5` で対戦が走る |
| P0 | エージェント開発者として、case5 を Kaggle に提出したい | `dev/submit rulebase/case5 --dry-run` が通り、本番提出も成功する |
| P0 | エージェント開発者として、notebook と同じ判断ロジックが移植できているか検証したい | 等価性 snapshot テスト + integration テストが通る |
| P1 | エージェント開発者として、各 mission の挙動を単体テストしたい | `tests/pipeline/rulebase/case5/` で各 mission builder の単体テストが書ける |
| P1 | エージェント開発者として、notebook 個別機能の効果を ablation で検証したい | `configs/baseline.yaml` + `evaluation/ablation.py` で個別機能 on/off を切り替えて 100-300 戦評価できる |
| P2 | エージェント開発者として、case4 と case5 の性能差を定量比較したい | `evaluation/compare_v2.py` (case4 と同等) で勝率を出せる |

## 3. 機能要件

### 3.1 移植する notebook 機能 (全機能を忠実移植)

| # | 機能 | notebook 参照 | 配置先 (case5) |
|---|------|--------------|---------------|
| F-01 | `Planet`/`Fleet` namedtuple, `ShotOption`/`Mission` dataclass | L179-205 | `core/types.py` |
| F-02 | 物理計算 (orbit_radius/predict_planet_position/predict_comet_position 等) | L208-385 | `core/physics.py`, `core/geometry.py` |
| F-03 | WorldModel (owner_strength/base_timeline/keep_needed_map/blood_in_water_owners 等) | L387-867 | `core/world_model.py` |
| F-04 | Strategy helpers (build_modes/target_value/reinforce_value/apply_score_modifiers/settle_plan) | L868-1504 | `strategy_helpers.py` |
| F-05 | capture mission | (build_capture_mission 相当) | `missions/capture.py` |
| F-06 | snipe mission (`build_snipe_mission`) | L1130-1200 | `missions/snipe.py` |
| F-07 | rescue mission (`build_rescue_missions`) | L1230-1290 | `missions/rescue.py` ★新規 |
| F-08 | recapture mission (`build_recapture_missions`) | L1290-1380 | `missions/recapture.py` ★新規 |
| F-09 | reinforce mission (`build_reinforce_missions`) | L1380-1460 | `missions/reinforcement.py` |
| F-10 | crash exploit mission (`build_crash_exploit_missions`) | L1460-1504 | `missions/crash_exploit.py` |
| F-11 | swarm (multi-source) ロジック (2源/3源切替) | L1700-1850 (in `plan_moves`) | `missions/swarm.py` |
| F-12 | proactive defense (`stacked_enemy_proactive_keep`) | L990-1050 | `movements/proactive_defense.py` ★新規 |
| F-13 | evacuation (太陽落下回避) | L1850-1900 (in `plan_moves`) | `movements/evacuation.py` |
| F-14 | rear guard (後方警備) | L1900-1950 (in `plan_moves`) | `movements/rear_guard.py` |
| F-15 | followup (連続着弾) | L1950-1985 (in `plan_moves`) | `movements/followup.py` |
| F-16 | exposed planets (`detect_exposed_enemy_planets` + `EXPOSED_PLANET_VALUE_MULT`) | L540-580, L900-920 | `core/world_model.py` (検出), `strategy_helpers.py` (適用) |
| F-17 | blood-in-water FFA 殲滅 | L600-640 | `core/world_model.py` (検出), `strategy_helpers.py` (適用) |
| F-18 | let-them-fight (敵 vs 敵中立争奪) | L580-600 | `core/world_model.py` (検出), `strategy_helpers.py` (ペナルティ適用) |
| F-19 | indirect features / apply_score_modifiers | L470-520, L1100-1130 | `core/world_model.py` + `strategy_helpers.py` |
| F-20 | mode 駆動の value 補正 (build_modes) | L1050-1100 | `strategy_helpers.py` |
| F-21 | deadline 制御 (`SOFT_ACT_DEADLINE` + `HEAVY_PHASE_MIN_TIME`) | L1986-2045 + 各重いフェーズ | `core/timing.py` ★新規 + `strategy.py` 内で活用 |
| F-22 | plan_moves オーケストレーション (collect → resolve → apply) | L1506-1985 (480行を分解) | `strategy.py` (~150行) |
| F-23 | agent エントリ (`_read`, `_detect_game_id`, `build_world`, `agent`) | L1987-2045 | `agent.py` |

### 3.2 移植しない機能

| # | 機能 | 理由 |
|---|------|------|
| N-01 | opponent_model.py | notebook に無い。memory `project_om_finding.md` で OM 効果なしと記録済 |
| N-02 | lookahead.py / predict_enemy_fleets | notebook に無い。OM 連動のため不要 |
| N-03 | harass mission | notebook に無い |

### 3.3 新規実装機能 (notebook 外)

| # | 機能 | 配置先 |
|---|------|------|
| F-24 | `core/timing.py` の `Deadline` dataclass + `should_skip(min_time)` helper | `core/timing.py` |
| F-25 | YAML ablation 設定 (`configs/baseline.yaml`) で各 mission on/off + 主要定数 override | `configs/baseline.yaml` |
| F-26 | `evaluation/ablation.py` / `compare_v2.py` / `snapshot_update.py` (case4 同等) | `evaluation/` |
| F-27 | `AGENT_REGISTRY` に `baseline_v5` 追加 | `src/dataset/selfplay/agents.py` |

### 3.4 非機能要件

| 項目 | 要件 |
|------|------|
| **時間制約** | 1 ターン 1.0s (`actTimeout=1.0`) 以内に必ず action を返す。`SOFT_ACT_DEADLINE=0.82s` を内部上限 |
| **構造制約** | backend.md 準拠: 関数 50 行以内、ファイル 200-400 行 (最大 800)、`Any` 禁止、`print` 禁止、frozen dataclass、型ヒント必須 |
| **ライセンス** | Apache 2.0 適用。`pipeline/rulebase/case5/baseline/LICENSE` に本文配置、`baseline/__init__.py` 冒頭で出典明記 |
| **可読性** | notebook の 480 行 `plan_moves` を 3 段階 (collect → resolve → apply) に分解、各 50 行以内 |
| **テスト性** | `core/timing.py` は monkeypatch で fake clock 注入可能。各 mission builder は WorldModel を渡せば単体実行可能 |
| **回帰検出** | snapshot test (固定 obs に対する action 一致) で `dev/test-backend` で回帰検出 |
| **互換性** | Python 3.13、`numpy`、`kaggle-environments` のみに依存。新規依存を追加しない |
| **対戦性能** | case4 (baseline_v4) との 100 戦自己対戦で **勝率 45% 以上** (case4 は LB 上位想定で同等性能の前提)。LB1224 notebook 由来であるため、notebook 側に劣ることは許容しない (snapshot test で同一 action 確認) |

## 4. スコープ外

| # | 項目 | 理由 |
|---|------|------|
| OOS-01 | 学習ベース戦略の追加 (RL/imitation) | rulebase 路線、別 case で扱う |
| OOS-02 | OM/lookahead の再導入 | notebook 路線と思想が異なる |
| OOS-03 | notebook と異なるパラメタチューニング | 移植段階では notebook と同一値を維持。チューニングは別 PR |
| OOS-04 | case4 のリファクタリング | case5 の独立実装、case4 は触らない |
| OOS-05 | UI / 可視化ツール | バックエンドエージェントのみ |

## 5. 用語

| 用語 | 説明 |
|------|------|
| Mission | 攻撃/救援/再奪還/補強/swarm/crash exploit など、艦隊 1 ペアの計画候補。dataclass `Mission(kind, source_id, target_id, angle, ships, score, eta)` |
| Mode | 局面フラグ (`is_ahead`, `is_behind`, `is_finishing`, `is_total_war`, `is_four_player`) を集約した `ModeFlags` |
| Deadline | `time.perf_counter()` ベースの絶対時刻。各重いフェーズ前に残時間チェックでスキップ判定 |
| Heavy phase | 計算重量の大きい mission builder (recapture/crash_exploit/3-source swarm) |
| Optional phase | 軽量だが省略可能な後処理 (followup, rear_guard) |
| Blood in water | FFA で艦数 ≤ `FFA_ELIMINATION_SHIPS=55` の弱った敵を検出する状態 |
| Exposed planet | 艦隊を出撃させて手薄になった敵惑星 |
