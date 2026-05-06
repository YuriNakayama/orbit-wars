# rulebase/case9 — anti_ping_pong

> 作成日: 2026-05-04
> 関連: `bot/pipeline/rulebase/case4/` (起点コピー元), `docs/experiment/rulebase/20260504_case7_accumulate_burst/iter1_plan.md` (ACCUMULATE 設計の参考)
> スコープ: Phase A 診断 → Phase B 余剰 ship の有効化。1 plan.md で 2 段構成、Phase B の詳細チューニングは iter2_plan.md で深掘りする想定。

## 仮説 (Hypothesis)

現行 rulebase (case4 = production) では、近接した友軍/敵惑星間で毎ターン小規模 ship が相互送出される **planet ping-pong** が発生している。

- **直接原因仮説 (H1)**: `missions/reinforcement.py` の `threatened_candidates` が `deficit_hint=1` (ship 1 でも fall_turn 検出) で発火し、隣接惑星から少量の補強艦を送る。送出元は ship 数が減った結果、自身が次ターン threatened 側に回り、相手から逆方向に補強される。
- **直接原因仮説 (H2)**: `missions/harass.py` が `HARASS_MIN_TARGET_SHIPS=1`, `needed+2` で発火するため、奪取 → 敵が同じロジックで奪い返す → 自分が奪い返す、の振動が起きる。
- **構造的原因仮説 (H3)**: `plan_moves(world)` は毎ターン履歴非依存で意思決定する。**「直前ターンに同じ src→dst pair で送った」を観測する hysteresis が存在しない**ため、最適性のごく僅かな揺れがそのまま振動になる。

→ ping-pong を抑止することで「縦割り少量送出」に消費されていた ship を保全し、(a) ACCUMULATE 風の多ターン蓄積 → 遠距離 1 発、(b) multi-source swarm の参加艦、(c) rear-guard reserve のいずれかに有効化することで、vs baseline_v4 の勝率改善 (+5pp 以上) を狙う。

## 既存コードの現状 (from Step 1)

- 起点は `bot/pipeline/rulebase/case4/` (production, vs v3 70.3%, publicScore 745)。`baseline_v4 = case3 + fleet_consolidation`。case8 は別ブランチで使用中のため、本実験は **新規 case9** を切る。
- 主要モジュール:
  - `case4/baseline/strategy.py`: `plan_moves(world)` が毎ターン mission を `collect_missions` → score 降順処理。**state は持たない**。
  - `case4/baseline/missions/reinforcement.py`: `threatened_candidates` を反復、`deficit_hint + REINFORCE_SAFETY_MARGIN(=5)` で発火。最小発火条件は実質 ships=1 から。
  - `case4/baseline/missions/harass.py`: `HARASS_MIN_TARGET_SHIPS=1`, `HARASS_MIN_TARGET_PRODUCTION=2`, `HARASS_MIN_SRC_RESERVE=10`, `HARASS_MAX_TRAVEL_TURNS=20` で 1 発奪取。
  - `case4/baseline/core/world_model.py:_compute_defense_buffers` (L516–561) が `threatened_candidates[planet.id] = {fall_turn, deficit_hint}` を生成。
  - `case4/baseline/agent.py`: モジュールレベルに `_OM_STATE: om.OMState` のみ持つ。**fleet 送出履歴の persistence は無し**。
- 過去 iter の所見:
  - `case7` (ACCUMULATE): 多ターン蓄積 → 遠距離 1 発。ping-pong 抑制が直接の動機ではないが、保全した ship の使い道として直接流用可能。
  - `project_case2_ablation`: 100 戦は seed variance 大、300 戦で評価。
  - `project_imitation_case1_phase3`: n<300 self-play は信頼不可。

## スコープ (Scope)

**Phase A: 診断 (iter1)**

- 新規ファイル:
  - `bot/pipeline/rulebase/case9/` を case4 のフルコピーで作成 (cross-case 独立性ルール準拠)。
  - `bot/pipeline/rulebase/case9/evaluation/diagnose_ping_pong.py` (新規、`.submitignore` 対象)。
- 変更ファイル: なし (Phase A は計測のみ)。
- 設定追加: なし。

**Phase B: 対策 (iter1 後半 → iter2 以降)**

- 変更ファイル:
  - `case9/baseline/core/config.py`: anti-ping-pong 関連定数を追加 (フラグで OFF 可)。
  - `case9/baseline/agent.py`: `_OM_STATE` と並んで `_DISPATCH_HISTORY` を保持する module-level state を追加。`build_world` で前ターンの dispatch 情報を `WorldModel` に注入。
  - `case9/baseline/core/world_model.py`: `WorldModel` に `recent_dispatches: dict[tuple[int,int], int]` (key: `(src_id, dst_id)` → 最終送信 turn) を保持。
  - `case9/baseline/missions/reinforcement.py`: `recent_dispatches` を見て同 pair が直近 N ターン以内に送信済みなら skip。`deficit_hint` 最小値も引き上げ。
  - `case9/baseline/missions/harass.py`: 同 target が直近 K ターン以内に harass 済みなら skip。
- ハイパーパラメータ追加 (config.py、すべて初期値はデフォルトで現状維持寄り、検証で調整):
  ```
  ANTI_PING_PONG_ENABLED: bool = True              # マスタフラグ。False で case9 ≡ case4
  PING_PONG_PAIR_COOLDOWN_TURNS: int = 3           # 同 src→dst pair の再送禁止 window
  HARASS_TARGET_COOLDOWN_TURNS: int = 5            # 同 target への harass 禁止 window
  REINFORCE_MIN_DEFICIT: int = 3                   # deficit_hint < この値なら threatened に入れない
  REINFORCE_MIN_TARGET_PRODUCTION: int = 2         # production 低い惑星の補強優先度を下げる (既存定数の流用検討)
  MIN_SHIPS_PER_LAUNCH: int = 4                    # 1 回の send が production×係数 未満なら抑止
  ```

**`AGENT_REGISTRY` 登録**

- `bot/src/dataset/selfplay/agents.py` に `"baseline_v9": "pipeline.rulebase.case9.baseline.agent:agent"` を追加 (実装ステップで実施)。

## 実装ステップ (Implementation outline)

### Phase A — 診断 (iter1 前半)

1. `bot/pipeline/rulebase/case9/` を `case4/` から複製 (relative import を保持)。`case9/README.md` を anti-ping-pong 用に書き直す。
2. `bot/pipeline/rulebase/case9/main.py` の sys.path / import 文は `Path.cwd()` パターンを継承 (case4 と同一)。
3. `bot/src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に `baseline_v9` を追加。
4. `bot/pipeline/rulebase/case9/evaluation/diagnose_ping_pong.py` を新規作成。仕様:
   - `kaggle_environments.make("orbit_wars")` で `agent` vs `agent` (self-play, 当面 baseline_v9 は case4 と等価) を N エピソード走らせる。
   - 各 step の action ([from_planet_id, angle, num_ships]) と planets を読み、各 fleet の **目的地 dst を `src + angle / planets` から推定** (`baseline.lookahead.predict_enemy_fleets` 周辺の helper を再利用、または近似で最寄り惑星を取る)。
   - **判定ルール**: 同じ `(src, dst)` または `(dst, src)` pair が `PING_PONG_WINDOW(=5)` ターン以内に逆方向送信を含めて 2 回以上発生したらカウント。
   - 出力: `data/output/diagnostics/ping_pong/{run_id}/summary.json` (pair, turn 帯, mission 推定 kind 別件数)、`top_pairs.parquet`。
5. `dev/test-bot` で format/lint/type/pytest が green を確認。`uv run --directory bot pytest tests/pipeline/rulebase/case9 -x` (case4 のテストを case9 にコピーしてパス先を書き換え)。
6. `uv run --directory bot python -m pipeline.rulebase.case9.evaluation.diagnose_ping_pong --episodes 300` を実行し、ping-pong 発生件数 / pair / turn 帯 / 主犯 mission を確認。
7. `iter1_result.md` 内の "Phase A 診断結果" 節に上記サマリを貼る。

### Phase B — 対策 (iter1 後半)

8. `case9/baseline/core/config.py` に上記 anti-ping-pong 定数を追加。
9. `case9/baseline/core/world_model.py` の `WorldModel` dataclass に `recent_dispatches: dict[tuple[int, int], int]` を追加 (default `{}`、immutable パターンを保つため builder 経由で注入)。
10. `case9/baseline/agent.py`:
    - `_DISPATCH_HISTORY: dict[tuple[int, int], int] = {}` を module-level に追加。
    - `agent(obs)` で `world.step` が `_OM_STATE.last_snapshot.step` よりも小さい場合は新規ゲームとして clear。
    - `build_world` で `recent_dispatches=_DISPATCH_HISTORY.copy()` を渡す。
    - `agent(obs)` の戻り値を確定後、`_DISPATCH_HISTORY` を更新 (`(src_id, est_dst_id) → world.step`)。dst 推定は `world.plan_shot` の逆引きで近似。
11. `case9/baseline/missions/reinforcement.py`:
    - 各 `(src, target)` ペアについて `world.recent_dispatches.get((src.id, target_id), -10**9)` を見て、`world.step - last < PING_PONG_PAIR_COOLDOWN_TURNS` なら `continue`。
    - `_compute_defense_buffers` 内 (world_model.py) の `threatened_candidates` 構築で、`deficit_hint < REINFORCE_MIN_DEFICIT` なら追加しない。
12. `case9/baseline/missions/harass.py`:
    - 同 target_id が `world.recent_dispatches` の値域 (`(_, target_id)` の最終 turn) を見て `world.step - last < HARASS_TARGET_COOLDOWN_TURNS` なら skip。
13. (任意) `case9/baseline/strategy.py` の `append_move` で `send < MIN_SHIPS_PER_LAUNCH and target.production >= 2` なら抑止 (production が高い惑星への少量送出のみ抑止)。production==1 の中立 capture は影響を受けない。
14. `dev/test-bot` を再実行。`pytest tests/pipeline/rulebase/case9 -x` で snapshot test を更新 (action 系列が変わるので新規 expected を作る)。

### Phase A/B の評価 (iter1 終盤)

15. `case9/evaluation/compare_v4.py` を新規 (case6 の `compare_*.py` パターン踏襲)。`baseline_v9 vs baseline_v4` を 300 戦走らせ、勝率 / draw / mean turn を JSON に出力。
16. 同時に `diagnose_ping_pong.py` を ANTI_PING_PONG_ENABLED=True/False で各 100 エピソード走らせ、ping-pong 件数の **絶対削減数 + 削減率** を `iter1_result.md` に記録。
17. 採否判定: 勝率 +5pp 以上、かつ ping-pong 件数 50% 以上削減、なら Phase B を採択。それ以外は iter2 で対策 (cooldown 値の調整 / 余剰 ship の振り向け先変更) を検討。

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format → lint → type → pytest)
  - `uv run --directory bot pytest tests/pipeline/rulebase/case9 -x`
  - `uv run --directory bot python -m submit submit rulebase/case9 --dry-run --skip-validation -m "case9 dry-run"`
- **diagnose スクリプト**:
  - `uv run --directory bot python -m pipeline.rulebase.case9.evaluation.diagnose_ping_pong --episodes 300 --enable-anti / --no-anti`
  - 出力先: `data/output/diagnostics/ping_pong/{run_id}/`
- **リモート**: RunPod 不要 (pure Python rule-based)。
- **評価**:
  - 対戦相手: **baseline_v4** (case4, production)。secondary に baseline_v6 (case6) と baseline_v7 (case7) を 100 戦ずつ参考測定。
  - エピソード数: **300** (`project_imitation_case1_phase3` 準拠)。
  - 主要メトリクス: **vs v4 勝率 + ping-pong 件数 (custom metric)**。
    - 勝率 ≥ 55% (= +5pp) を採択しきい値。
    - ping-pong 件数 (= 同 pair が cooldown window 以内に逆方向送出された回数) は **減少することを補助確認**として要求 (50% 以上削減目安)。
  - Kaggle publicScore は **採否根拠としない** (memory `project_om_finding`, `project_case5_validation` 準拠)。
- **採否しきい値**:
  - **採択**: vs v4 勝率 ≥ 55% (300 戦) かつ ping-pong 件数 50% 以上削減。
  - **保留 (iter2)**: 勝率 50–55%、または件数削減はあるが勝率改善が乏しい → cooldown 値や `MIN_SHIPS_PER_LAUNCH` を tune。
  - **棄却**: 勝率 < 50%、または件数は減ったが勝率明確に低下 → 余剰 ships の使い道を ACCUMULATE 連携に切り替えて iter2_plan.md で再設計。

## 余剰 ship の有効化方針 (iter2 以降への引き継ぎ)

ユーザー希望に基づき、Phase A 診断結果を見て iter2_plan.md で詳細化する。優先度高い順:

1. **ACCUMULATE 風蓄積 → 遠距離 1 発**: case7 の `ACCUMULATE_ENABLED` 系定数と mission を case9 にも導入し、抑止された ship を蓄積側に回す。case7 の実装をそのまま port するため変更コスト低。
2. **Multi-source swarm 優先割当**: `swarm.py` に `prefer_anti_ping_pong_sources` フラグを足し、recent_dispatches の少ない src を優先。
3. **Rear-guard reserve 強化**: `movements/rear_guard.py` の `REAR_SOURCE_MIN_SHIPS` を動的に下げて、抑止された ship を後方惑星に rotate。

iter1 で診断結果が出てから、どれを最初に組み込むかを iter2_plan.md で決定する。

## 想定リスク

- **dst 推定の誤差**: `recent_dispatches` の dst は `plan_shot` 角度から逆引き推定なので、複数の候補惑星が同方向にある場合に誤判定する。Phase A の diagnose スクリプトで誤判定率を確認し、許容できなければ「`(src_id, angle bucket)` ベース」に切り替える。
- **過抑止による初動遅延**: cooldown が長すぎると正当な reinforce / harass まで止まり、序盤に押し負ける可能性。case4 比で序盤 30 ターンの enemy ships 増加率を diagnose スクリプトで併測。
- **case8 不在による評価漏れ**: case8 (別ブランチ) との比較は本 plan 外とする。必要なら別 worktree で別途実施。
