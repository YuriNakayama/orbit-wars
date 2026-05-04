# [rulebase/case7] accumulate-then-burst: 多ターン蓄積からの遠距離単発攻撃

> 作成日: 2026-05-04
> 関連:
> - `docs/experiment/rulebase/20260502_case6_stay_mission/iter7_result.md` (case6 確定構成 cap=3, 300戦 vs v5 で 59.7%)
> - `docs/experiment/rulebase/20260502_case6_stay_mission/iter5_result.md` (case6 vs v5 cap=3 breakthrough)
> - `bot/pipeline/rulebase/case6/baseline/missions/stay.py` (1ターン burst 実装)
> スコープ: case6 を fork して `case7/baseline/missions/stay.py` の burst hold を多ターン拡張、accumulation を mission として scoring に乗せる。

## 仮説 (Hypothesis)

**敵脅威スコアが低い友軍惑星で ship 数を「目標惑星捕獲に必要な ships + safety + fleet_speed knee」に達するまで複数ターン蓄積し、揃った時点で遠距離の友軍 / 敵惑星に単発攻撃する mission を導入することで、case6 (1ターン arbitrage) を上回る fleet 形成と命中率が得られる**。

理由 (Mechanism):
- `fleet_speed = 1 + (max-1) * (log(ships)/log(1000))^1.5` は ships が大きいほど移動速度が増し、knee (~316 ships, log10 ≈ 2.5) で速度カーブが飽和する。
- case6 は「1 ターン待つと ETA が縮むなら hold」という arbitrage で +5pp 改善 (300戦 vs v5)。しかし iter7 で `MAX_HOLD_TURNS>=5` がプラトーに頭打ちなのは、**1 ターン単位の局所判断では「最終的に必要な ships 量」までの長期蓄積を許容できない**ため。
- 「捕獲必要量 + safety + knee 近傍」をしきい値とする target-aware accumulation phase なら、文脈喪失 (case6 で観測された stuck holds) を起こさず、**遠距離の高価値惑星に対して単発で勝負を決める** 形に近づく。

## 既存コードの現状 (from Step 1)

- 主要モジュール: `bot/pipeline/rulebase/case6/baseline/`
  - `strategy.py` (`plan_moves`) — `collect_missions` → score 順処理 → `emit_followup/evacuation/rear_guard_moves` の 3 段。STAY hold は `build_stay_holds` で per-source の `attack_left` から減算。
  - `missions/stay.py` (`build_stay_holds`) — defense (現在 OFF) + burst の 2 系統。burst は `_best_burst_target` で 1 ターン待ちの ETA gain `eta_now - (eta_next + 1) >= 1` を判定。
  - `core/config.py` の STAY 系: `STAY_BURST_MIN_GAIN=1`, `MIN_SHIPS=8`, `MAX_TARGET_TURNS=30`, `MAX_HOLD_TURNS=3` (cap=3 が iter5 で確定)。
  - `core/world_model.py` — `ships_needed_to_capture(target, arrival_turn, planned_commitments)` で必要量算出可能、`cached_travel_time` で ETA 計算可能。
- `bot/src/dataset/selfplay/executor.py` 100行目: `import orbit_wars_rust` 副作用 import で kaggle_environments の orbit_wars を Rust backend に切替。`compare_v5.py` 経由なら自動で Rust 評価になる。
- 過去 iter の所見:
  - cap=3 で plateau (cap=4/5 同等、cap=2 急落、cap=6 劣化、cap=∞ で文脈喪失害顕在化)。
  - `MAX_TARGET_TURNS=25` (-5) で勝率 -7pp、長距離 hold は本質的に勝率寄与。
  - iter7 結論: 「case6 の 1 ターン arbitrage では伸び代終了。次は新機構 (multi-turn accumulation, defense+burst 2段, comet-aware) で case7」。

## スコープ (Scope)

- 新 case 雛形:
  - `bot/pipeline/rulebase/case7/__init__.py`
  - `bot/pipeline/rulebase/case7/main.py` (`Path.cwd()` 方式の sys.path 注入、`from baseline.agent import agent`)
  - `bot/pipeline/rulebase/case7/README.md` (case6 から派生、accumulate-burst 戦術概要)
  - `bot/pipeline/rulebase/case7/baseline/` 一式 — case6 から **コピー** (cross-case import 禁止ルール準拠)。
- 変更ファイル (case7 内):
  - `baseline/missions/stay.py` — `_best_burst_target` を多ターン累積版に拡張、target ship しきい値 `target_threshold(target, world)` を導入。
  - `baseline/missions/__init__.py` / `baseline/strategy.py` — accumulation を mission として `collect_missions` に登録 (kind="accumulate") し、score 比較で他 mission と競合させる。
  - `baseline/core/config.py` — case7 専用の以下を追加:
    - `ACCUMULATE_ENABLED: bool = True`
    - `ACCUMULATE_THREAT_MAX: float` (蓄積を許可する敵脅威スコア上限)
    - `ACCUMULATE_SAFETY_SHIPS: int` (capture 必要量への上乗せ)
    - `ACCUMULATE_KNEE_SHIPS: int = 60` (fleet_speed knee 近傍の参考値、log curve で実測決定)
    - `ACCUMULATE_MAX_HOLD_TURNS: int` (新 cap、case6 cap=3 とは独立に sweep 候補)
    - `ACCUMULATE_MIN_TARGET_TURNS: int = 15` (近距離は STAY/通常 mission に任せる)
- ハイパーパラメータ初期値 (config 上の before → after):
  - `STAY_BURST_*` 系は case6 と同値で固定 (1ターン arbitrage は維持)。
  - `ACCUMULATE_*` は新規追加。初期値は `THREAT_MAX=0.3`, `SAFETY=4`, `KNEE=60`, `MAX_HOLD=8`, `MIN_TARGET=15` を仮置き、Stage1 で sweep。
- データセット / 特徴量変更: なし (rulebase は学習なし)。
- AGENT_REGISTRY 登録: `bot/src/dataset/selfplay/agents.py` に `"baseline_v7": "pipeline.rulebase.case7.baseline.agent:agent"` を追加。
- 評価スクリプト: `case7/evaluation/compare_v6.py` を `case6/evaluation/compare_v5.py` を雛形に新規追加。

## 実装ステップ (Implementation outline)

1. **case7 雛形作成**: `bot/pipeline/rulebase/case7/` を `case6` から `cp -r` し、`baseline/` 内の絶対 import が無いこと (相対 import のみ) を確認。`README.md` を case7 用に書き直し。
2. **AGENT_REGISTRY 登録**: `bot/src/dataset/selfplay/agents.py` に `baseline_v7` を追加。
3. **しきい値計算ヘルパ**: `case7/baseline/missions/stay.py` に `_accumulate_target_threshold(target, src, world, planned_commitments) -> int` を新設。
   - `world.ships_needed_to_capture(target.id, arrival_turn, planned_commitments)` で capture 必要量取得。
   - `+ ACCUMULATE_SAFETY_SHIPS`、`max(_, ACCUMULATE_KNEE_SHIPS)` で knee 下回りを除外。
4. **多ターン accumulate mission の生成**: `case7/baseline/missions/accumulate.py` (もしくは stay.py 内に kind 拡張) で:
   - 敵脅威スコア (`world.threat_score` 相当、無ければ `defense_risk_for_planet` を流用) が `ACCUMULATE_THREAT_MAX` 未満の友軍 source のみ候補。
   - 各 (src, target) ペアで:
     - 距離が `ACCUMULATE_MIN_TARGET_TURNS` 以上 (近距離は STAY/通常 mission に任せる)。
     - 現在 `usable_ships < threshold` なら hold (mission として `kind="accumulate"`, score = `(threshold - usable_ships) ^-1 × target_value × speed_gain`)。
     - `usable_ships >= threshold` なら mission として fire (kind="accumulate_fire") を `collect_missions` に登録、`score` で他 mission と競合。
5. **strategy.py 拡張**: `_process_single_source_mission` の `SINGLE_SOURCE_MISSION_KINDS` に `"accumulate_fire"` を追加。`accumulate_hold` は `build_stay_holds` の merge と同様に per-source `attack_left` を 0 にする扱いを追加。
6. **MAX_HOLD_TURNS 累積**: case7 では accumulation phase を超えて hold した連続ターン数を `consecutive_holds` 同様に追跡し、`ACCUMULATE_MAX_HOLD_TURNS` で stuck-hold を防止。
7. **テスト追加**: `bot/tests/pipeline/rulebase/case7/` に以下を追加:
   - `test_accumulate_threshold.py` — 必要量 + safety + knee の式が config 値で正しく動くか。
   - `test_accumulate_fires.py` — usable_ships が threshold を超えたターンに fire mission が emit されるか。
   - `test_accumulate_hold_cap.py` — `MAX_HOLD_TURNS` 超過で強制発火 or 別 mission への譲渡が起きるか。
   - `test_accumulate_priority.py` — 敵脅威スコアが高い source は accumulate せず defense / 通常 mission に譲るか。

## 検証方法 (Validation method)

- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/rulebase/case7 -x`
- (submit-shape change の場合) `uv run --directory bot python -m submit submit rulebase/case7 --dry-run --skip-validation -m "case7 dry run"` (Path.cwd 方式の確認のみ、実提出はしない)。
- リモート: **不要** (rulebase は GPU 不使用、RunPod は使わない)。
- シミュレータ: **Rust backend** を使う。`bot/src/dataset/selfplay/executor.py` の副作用 import が `case7/evaluation/compare_v6.py` 経由で自動切替されることを `nvidia-smi` 不在環境でも確認 (`orbit_wars_rust` ModuleNotFoundError 時は Python fallback の警告だけ出る)。
- 評価:
  - 対戦相手: **baseline_v6** (case6 cap=3 確定構成)。`AGENT_REGISTRY` に登録済の名称で参照。
  - エピソード数: **300戦** (memory `project_imitation_case1_phase3` の n<300 不可ルール準拠)。
  - 主要メトリクス: ローカル自己対戦の勝率 (case6 と同形式)。Kaggle publicScore は使わない (memory `project_om_finding`, `project_case5_validation` 準拠)。
  - 副次メトリクス: fleet peak ratio (case7/case6)、平均 launches/episode、hold→fire 達成率 (accumulate phase に入った source のうち fire まで到達した割合)、平均 hold turns。
  - 採否しきい値: **勝率 ≥ 53%** (= +3pp 以上、二項検定で n=300, p=0.5 のとき p ≈ 0.16 であり「弱い証拠」だが、ユーザー指定の小さな伎いも採用方針に従う)。**先に 100戦 quick で勝率 < 50% なら早期却下**。
  - Stage 構成:
    - **Stage 0 (smoke)**: 100戦 vs baseline_v6 (12 物理コアで 3 並列、~20分)。50% を著しく下回る場合は config 初期値が不適合と判断し sweep に切替。
    - **Stage 1 (sweep)**: `THREAT_MAX ∈ {0.2, 0.3, 0.5}`, `SAFETY ∈ {2, 4, 8}`, `MAX_HOLD ∈ {5, 8, 12}` を 3軸 × 100戦 (合計 ~3 時間)。最良点を選定。
    - **Stage 2 (confirm)**: 最良 config で 300戦 vs baseline_v6 (~1 時間)。**勝率 ≥ 53% で採用**、< 50% で棄却、50〜53% は seed 別 100 戦追加で再判定。

## 参考 (References)

(Step 3 で web 調査不要と判定したため、本セクションは省略)
