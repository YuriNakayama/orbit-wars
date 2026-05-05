# Rulebase/case8 — Multi-step Beam Search Optimizer

> 作成日: 2026-05-04
> 関連:
> - `docs/experiment/rulebase/20260420_case3_rollout_ablation/result.md` (本実験は case3 result.md が「次の改善には MCTS / beam search が必要」と明記した方向の続編)
> - `docs/experiment/rulebase/20260504_case7_accumulate_burst/iter1_result.md` (base となる case7 の最新所見)
> スコープ: case7 を base として複製し、`baseline/planner/` サブパッケージで greedy mission selector を multi-turn beam search optimizer に置換する

## 仮説 (Hypothesis)

`strategy.plan_moves` の greedy mission selector (mission を score 降順に 1 個ずつ commit) は、(a) ターン内のミッション間で艦数バジェットを取り合った時の **joint 最適性**、(b) 自軍の発射が **次ターン以降の盤面** に与える影響の **2 軸** で局所解に陥っている。これを **multi-turn beam search** (横断 N=2-3 ターン、beam_width B=4-8) に置換することで、vs `baseline_v4` (production) のローカル 300 戦勝率を **+5pp 以上 (≥55%)** 改善できる。

**Mechanism / なぜ効くと期待できるか**:
- case3 result.md の教訓: `score 補正 / top-K reorder` 系 (OM v1/v2, lookahead 1-ply, ROLLOUT bonus/replace/filter, true2p) は **5 連敗** で seed variance を破れず、heuristic score と rollout 値が強く相関しているため top-K 内の並び替えだけでは挙動が有意に変わらない。同 result.md は「**MCTS / beam search (shallow rollout ではなく全候補から選抜する探索)**」を未踏領域として明示。
- 本実験は (a)「候補生成そのものは既存 missions を流用しつつ」、(b)「組合せ選抜と多ターン展開を beam で同時に行う」ことで、score 補正系で飽和した改善余地を別軸で開拓する。RTS 文献 (Portfolio Greedy Search, Nested-Greedy) と整合する戦略。

## 既存コードの現状 (from Step 1)

- **base case**: `bot/pipeline/rulebase/case7/` (`baseline_v7`, `accumulate_burst`)
  - `baseline/agent.py` (173行): WorldModel 構築 + OM 更新 + lookahead 注入 → `plan_moves` 呼び出し
  - `baseline/strategy.py` (261行) `plan_moves`: `collect_missions` → `missions.sort(-score)` → 上から `_process_single_source_mission` / `_process_multi_source_mission` で commit (line 216-247)、その後 `emit_followup_moves` / `emit_evacuation_moves` / `emit_rear_guard_moves`
  - `baseline/missions/` (8 種: capture, snipe, reinforce, harass, swarm, crash_exploit, accumulate_fire, stay)
  - `baseline/movements/` (3 種: evacuation, followup, rear_guard)
  - `baseline/lookahead.py` (143行): **敵 fleet 予測のみ**、自軍 action 探索なし
  - `baseline/opponent_model.py`: 既存 OM v1/v2 (default OFF; メモリ `project_om_finding` 通り)
- **既存の探索系資産**: `case3/baseline/rollout.py` の `simulate_planet_timeline`, `_baseline_net_ships`, true2p の `plan_moves(world, light=True)` 経路 — case8 にコピーして再利用予定
- **AGENT_REGISTRY**: `bot/src/dataset/selfplay/agents.py:23` に `baseline_v7` まで登録済み。`baseline_v8` を追加する
- **過去 iter の所見** (case3 result.md, 2026-04-20):
  - score 補正系は 5 連敗で seed variance に埋没 → 別軸が必要
  - true2p (300戦 51.7%, seat 対称 ±0.35pp) が「seat 対称性で唯一信頼できる挙動安定」 → 本実験の評価関数として再利用可能
  - TOP_K=8 + light で turn_p95 0.75-0.82s に達し timeouts 5件 → **0.7s が安全帯の上限**

## スコープ (Scope)

### 変更ファイル

```
bot/pipeline/rulebase/case8/                                # case7 を全複製
├── __init__.py
├── main.py                                                  # case7 と同型 (Path.cwd() 系 sys.path injection)
├── README.md                                                # baseline_v8 概要
├── baseline/
│   ├── __init__.py
│   ├── agent.py                                             # case7 から複製、plan_moves の呼び出しは互換維持
│   ├── strategy.py                                          # ★greedy ループを planner.beam.run へ delegate
│   ├── strategy_helpers.py                                  # 既存維持 (build_modes / preferred_send)
│   ├── opponent_model.py                                    # 既存維持
│   ├── lookahead.py                                         # 既存維持
│   ├── core/                                                # 既存維持
│   ├── missions/                                            # 既存維持 (mission 列挙ロジック)
│   ├── movements/                                           # 既存維持 (followup/evacuation/rear_guard)
│   └── planner/                                             # ★新設サブパッケージ (case1 の planner/ 構造を参考)
│       ├── __init__.py                                      # `from .beam import run as run_beam`
│       ├── beam.py                                          # ★beam search core (~150-200行想定)
│       ├── candidate.py                                     # mission 部分集合 → MovesPlan の生成・列挙
│       ├── evaluator.py                                     # plan 評価関数 (case3 流用 + multi-turn 拡張)
│       └── simulator.py                                     # case3 rollout.py から simulate_planet_timeline 系を抽出複製
├── configs/                                                 # 既存維持 (BeamConfig dataclass 用 yaml 1 枚を新規追加)
└── evaluation/                                              # 既存維持 (compare_v8_vs_v4.py を新規追加)

bot/src/dataset/selfplay/agents.py                           # `"baseline_v8": ...` を追加
bot/tests/pipeline/rulebase/case8/                           # ★新設、case7 のスナップショット系テストを複製 + planner 単体テスト
```

**重要**: cross-case import 禁止 (`.claude/rules/bot/pipeline.md`) のため、`case3/baseline/rollout.py` の有用関数は **コピーしてくる** こと。共通化はしない。

### ハイパーパラメータ / config 変更 (`baseline/core/config.py` に追加)

| 名前 | デフォルト | 変動範囲 (tuning 候補) | 役割 |
|------|----------|----------------------|------|
| `BEAM_ENABLED` | `True` | True/False | OFF で case7 と完全同等動作 (regression check に使用) |
| `BEAM_HORIZON` | `2` | 1-3 | 何ターン先まで simulate するか |
| `BEAM_WIDTH` | `4` | 2-8 | 各 depth で残す上位 plan 数 |
| `BEAM_BRANCH_LIMIT` | `8` | 4-12 | 1 plan あたりに展開する mission 部分集合数 (top-N 候補ミッション + sample) |
| `BEAM_OPPONENT_MODE` | `"static"` | "static"/"true2p_light" | 敵を静止と仮定するか、敵 plan_moves(light) を使うか |
| `BEAM_VALUE_WEIGHTS` | `(1.0, 0.5, 0.3)` | 各 ±0.5 | (自軍純艦数, 自軍 production 合計, 敵 home 脅威スコア) の重み |
| `BEAM_TIME_BUDGET_S` | `0.6` | 0.4-0.8 | 1 ターンあたりの beam 探索時間上限 (超えたら early-stop で best-so-far 採用) |

「OFF で完全同等動作」を保つことで、greedy → beam の差分のみが勝率変化として観測できる。

### データセット / 特徴量変更

なし (ルールベースのため学習データ不要)。

## 実装ステップ (Implementation outline)

1. **case7 全複製** — `cp -r bot/pipeline/rulebase/case7 bot/pipeline/rulebase/case8` した上で、`README.md` / `__init__.py` の参照を case7 → case8 に置換、Kaggle entrypoint コメントも更新。
2. **`baseline/planner/simulator.py`** — `case3/baseline/rollout.py` から `simulate_planet_timeline`, `_baseline_net_ships`, `_infer_action_target`, `_strongest_enemy`, `_enemy_reaction_arrivals` をコピー (cross-case import せず複製)。`WorldModel` を入力として **N ターン進めて** 自軍純艦数 + production 合計 + 敵脅威スコアを返す `evaluate_plan(world, plan, horizon)` を新規実装。
3. **`baseline/planner/candidate.py`** — `MovesPlan` dataclass (= `list[move]` + `planned_commitments` + `mission_ids`) を定義。`enumerate_branches(world, base_plan, top_k=BEAM_BRANCH_LIMIT)` で「未採用ミッションから 1 個追加 / 既採用ミッションを 1 個 drop / 何もしない」の 3 方向に分岐する。
4. **`baseline/planner/evaluator.py`** — `score_plan(world, plan, weights, horizon, opponent_mode)` を実装。`opponent_mode="static"` なら predicted_arrivals 据え置き、`"true2p_light"` なら `enemy_arrivals = _enemy_reaction_arrivals(world, ...)` を folding。
5. **`baseline/planner/beam.py`** — `run(world, missions, modes, time_budget_s, beam_width, horizon, branch_limit) -> MovesPlan` を実装。
   - 初期 beam = `[empty_plan]`
   - 各 depth d=0..horizon-1 で: 全 plan を `enumerate_branches` で展開 → `score_plan` で評価 → 上位 `beam_width` 個を残す
   - `time.perf_counter()` で `time_budget_s` を超えたら現状の最良を返す (early stop)
   - 返り値は depth=0 で実行する moves のみ (depth>0 の plan は探索のためだけに使用)
6. **`baseline/strategy.py`** — `plan_moves` の頭で `if BEAM_ENABLED: return planner.beam.run(...)`。followup/evacuation/rear_guard は既存通り beam の出力に対して append。`plan_moves(world, light=True)` の経路 (敵 reaction 用) は **beam を呼ばず greedy** にする (再帰防止 + 軽量化)。
7. **`bot/src/dataset/selfplay/agents.py`** — `"baseline_v8": "pipeline.rulebase.case8.baseline.agent:agent"` を `AGENT_REGISTRY` に追加。
8. **`bot/tests/pipeline/rulebase/case8/`** — case7 のスナップショットテストを複製 + 以下を追加:
   - `test_beam_off_equals_greedy.py` — `BEAM_ENABLED=False` で case7 と同じ moves を返すこと
   - `test_beam_time_budget.py` — `BEAM_TIME_BUDGET_S=0.05` でも合法 moves を返すこと (early stop 機能確認)
   - `test_planner_evaluator.py` — `score_plan` の単体テスト
9. **`pipeline/.submitignore`** — `evaluation/`, `configs/` は既に除外されているため追加対応不要 (case 共通)。

## 検証方法 (Validation method)

### ローカル

```bash
# 必須: 形式チェック + lint + type + 既存 + 新規 unit test 全通し
dev/test-bot

# case8 単独テスト (高速ループ用)
uv run --directory bot pytest tests/pipeline/rulebase/case8 -x

# beam off=greedy regression
uv run --directory bot pytest tests/pipeline/rulebase/case8/test_beam_off_equals_greedy.py -x

# submit-shape は変えない (mission 構造ごと再利用) ため --dry-run は省略可。ただし新規 case 投入時は必ず:
uv run --directory bot python -m submit submit rulebase/case8 --dry-run --skip-validation -m "case8 dry-run"
```

### 性能評価 (採否判定の本体)

```bash
# vs baseline_v4 (production) 300戦、seat 入替で対称性も確認
uv run --directory bot python -m env run --agents baseline_v4,baseline_v8 --mode 1v1 -n 150 --seed 50000 --parallel 4
uv run --directory bot python -m env run --agents baseline_v8,baseline_v4 --mode 1v1 -n 150 --seed 50500 --parallel 4
```

- **対戦相手**: `baseline_v4` (production, LB745) を主軸。補助で `baseline_v7` (case8 の base) との比較で「beam の純粋寄与」を見る。
- **エピソード数**: 合算 300 戦 (seat0=150, seat1=150)。case3 の経験で n<300 は seed variance に埋没するため。
- **主要メトリクス**: 合算勝率 (vs v4)。**Kaggle publicScore は使用しない** (memory `project_om_finding` / `project_case5_validation` の通り、opponent pool drift で信頼不可)。
- **採否しきい値**: 合算 **+5pp 以上 (≥55%)** で採用。case3 の +1.3pp が seed variance 内だった経験から、+5pp で初めて「greedy に対する有意な改善」と扱える。
- **Time budget**: `turn_p95 ≤ 0.7s` (`actTimeout=1.0s` の 70%)。case3 H-a (0.87s で timeouts 61件) を踏まえた安全帯。0.7s を超える構成は採用候補から外す。

### リモート

**RunPod 不要**。ルールベースのため GPU 学習なし。tuning は CPU 並列 (`--parallel 4`) のローカル 300 戦をスイープで実施。

### Tuning 計画 (合計 ~5 構成)

1. **baseline (`BEAM_HORIZON=2`, `BEAM_WIDTH=4`, `BEAM_OPPONENT_MODE="static"`)** — 最低限の beam で greedy を破れるか
2. **+horizon=3** — 深さの効果単独
3. **+width=8** — 幅の効果単独
4. **+opponent_mode="true2p_light"** — 敵 reaction を入れる (case3 G の延長)
5. **best-of-1〜4 + value weights tuning** — 上位構成で重みを ±0.5 振る

各構成 vs v4 300戦。turn_p95 が 0.7s を超えた構成は即時棄却。

## リスクと早期撤退条件

- **Time budget 破綻**: turn_p95 が 0.7s を超え、`branch_limit` / `width` / `horizon` のどれを下げても 0.5s を切れない → 構成 1 段階下げて再評価。それでも greedy に勝てなければ撤退。
- **Seed variance 埋没**: 合算 300 戦で +3pp 未満 → case3 と同じ「score 補正系の飽和」パターン。撤退して `result.md` に「heuristic ベースの multi-step 化も飽和」と記録。次の方針 (学習ベース評価関数 / MCTS) に進む判断材料とする。
- **Seat 非対称 ±10pp 超**: case3 の D 施策と同じ挙動バグ警告。原因特定まで採用保留。

## 参考 (References)

- [Portfolio Search and Optimization for General Strategy Game-Playing (arxiv 2104.10429)](https://arxiv.org/pdf/2104.10429) — Portfolio Greedy Search (PGS) は script 集合の組合せを hill climbing で探索する RTS AI 標準手法。本実験の「既存 mission を script として扱い、その部分集合を beam で選抜する」構成と直接対応する理論基盤。
- [Nested-Greedy Search for Adversarial Real-Time Games (Moraes, AAAI)](https://cdn.aaai.org/ojs/13017/13017-52-16534-1-2-20201228.pdf) — 1-step greedy → nested-greedy への拡張は本質的に多重ループの depth=2 探索。本実験の `BEAM_HORIZON` 設計に対応。
- [Parametric Action Pre-Selection for MCTS in RTS Games (CEUR Vol-2719/paper11)](https://ceur-ws.org/Vol-2719/paper11.pdf) — 大きな combinatorial action space を heuristic で pre-select してから MCTS / beam を回す設計。`collect_missions` → beam の流れの妥当性を裏付ける。
- [Planet-Wars/InferBot.cpp (9thbit, GitHub)](https://github.com/9thbit/Planet-Wars/blob/master/InferBot.cpp) — 2010 年 Google AI Challenge Planet Wars の lookahead 戦略実装例。「自分が動く → 敵が反応する」という 1.5-ply 構造は case3 true2p で既に実装済み。本実験では beam 内の 1 ステップとして取り込む。
