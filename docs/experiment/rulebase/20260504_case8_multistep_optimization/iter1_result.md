# Rulebase/case8 — Multi-step Beam Search Optimizer (Result)

> 作成日: 2026-05-04
> 対応 plan: [`plan.md`](./plan.md)
> 関連:
> - [`docs/experiment/rulebase/20260420_case3_rollout_ablation/result.md`](../20260420_case3_rollout_ablation/result.md) — score 補正系が seed variance に埋没、「次は MCTS / beam」と明記した先行調査
> - [`docs/experiment/rulebase/20260504_case7_accumulate_burst/iter1_result.md`](../20260504_case7_accumulate_burst/iter1_result.md) — case8 の base となる case7 の最新所見

## 結論

**仮説は否定。case8 は採用却下。**

vs `baseline_v4` (production, LB745) の 300 戦合算で **win_rate 32.3% (-17.7pp from neutral)**。
plan.md のしきい値「+5pp 以上 (合算 ≥55%)」を **大幅に下回り**、純粋な後退。
seat 対称性は ±1.65pp で **構造的バグではなく挙動由来の劣化**。
turn_p95 = 0.310s で time budget には十分余裕があり、計算予算が原因ではない。

## 数値

### 主要メトリクス: vs baseline_v4 (case4, production)

| seat | n | baseline_v4 wins | baseline_v8 wins | v8 win_rate | v4 turn_p95 | v8 turn_p95 | timeouts |
|------|---|---|---|---|---|---|---|
| seat A (v4 first, seed 50000+) | 150 | 104 | 46 | 30.7% | 0.880s | 0.310s | 0 |
| seat B (v8 first, seed 50500+) | 150 | 99 | 51 | 34.0% | 0.616s | 0.305s | 0 |
| **合算** | **300** | **203** | **97** | **32.3%** | — | **0.31s** | **0** |

seat 対称性: ±1.65pp (50% を中心に対称)。case3 result.md で問題視された
seat 非対称 (±13pp) は再発しておらず、構造的な seat-bug ではない。

### 補助観察 (smoke 10 戦)

| 対戦 | n | v8 win_rate | 備考 |
|------|---|---|---|
| v4 vs v8 | 10 | 20.0% | 合算 32.3% と整合 (n=10 はノイズ過多だが方向は同じ) |
| v7 vs v8 | 10 | 50.0% | beam は base case7 に対しほぼニュートラル |
| v4 vs v7 | 10 | 40.0% | **case7 自体が v4 に劣る** — case8 劣勢の主因の一部 |

### しきい値判定

| 項目 | plan.md しきい値 | 実測 | 判定 |
|------|----------------|------|------|
| 合算勝率 vs v4 | ≥55% (greedy 比 +5pp) | 32.3% | ❌ -22.7pp 未達 |
| seat 対称性 | ±10pp 未満 | ±1.65pp | ✅ |
| turn_p95 | ≤0.7s | 0.31s | ✅ |
| timeouts | 0 件 | 0 件 | ✅ |

採用判定の主条件 (合算勝率) で 22.7pp の不足。**撤退**。

## 診断 — なぜ negative だったか

### (1) 単純な「greedy より弱い ordering を選んでいる」

beam search は score-desc greedy ordering を seed として持ち、それより
**評価関数 `score_commitments` が高く出た ordering を採用** する。
評価関数は `simulate_planet_timeline` で全惑星を horizon=2 ターン展開し
線形和 `1.0 * net_ships + 0.5 * player_production - 0.3 * enemy_threat` を
返す。

仮説: heuristic の `mission.score` (capture/snipe/reinforce/swarm/harass の
合算優先度) と、horizon-end の `net_ships` 主体の評価関数は **相関するが
一致しない**。greedy が「即時の局所最適」を選ぶのに対し、beam の評価関数は
「2 ターン先の総艦数」を選び、これが対人戦では **守備手薄化** や **重複発射**
に繋がっている可能性が高い。case3 result.md が同種の score 補正系で
5 連敗した教訓 (heuristic と rollout 値の相関で並び替え意味なし) と
**逆の現象**: ここでは相関が崩れて並び替えが「悪い方向」に効いている。

### (2) horizon=2 / 静的敵モデル

`BEAM_OPPONENT_MODE="static"` のため敵反応を folding せずに評価。
2 ターン先の自軍純艦数を最大化する ordering は、敵反撃を受けると
艦数が逆転する局面でも採用されやすく、ここが致命傷の可能性がある。
case3 G (true2p) の知見「敵反撃を入れた 51.7%」は本実験でも適用すべき
だったが、まずは static で計測する設計だった。

### (3) base case7 の v4 に対する弱さ

smoke で観察された v4 vs v7 = 60-40 から、case7 自体が v4 に劣る
(case7 = case6 + accumulate; LB1224 port の case5 系資産は持たない)
構造的弱点があり、case8 はその上に beam の負荷を載せたため二重に劣化した。
仮に beam の評価関数が完璧でも、**base 選定が誤り** だった可能性。

## 採用方針

- **case8 は採用却下** (棄却)
- `bot/src/dataset/selfplay/agents.py` の `baseline_v8` 登録は **保持** (
  ablation や後続の改修で再利用可能、コストなし)
- `BEAM_ENABLED` flag は **default `True` のまま** だが、case8 自体を
  Kaggle に submit しない方針なので影響なし。次の iter で改修する場合は
  この flag を握って ablation を継続できる
- production 候補は引き続き **case4 (baseline_v4, LB745)** のまま

## 次の iter で試す価値があるもの (棄却ではなく方向転換)

case3 result.md と本実験の 2 連敗で、**「heuristic score の上に評価関数や
beam を重ねる」方針** はおそらく飽和している。次は構造的に異なる軸:

1. **base を case4 に変更** — 同じ beam 構造を case4 (production) の
   greedy に被せて測定し、beam が「より強い base」上でも害を成すかを切り分ける。
   1 戦あたりの計算量は同じなので低コスト。
2. **評価関数を `mission.score` 基準に変更** — `score_commitments` を
   「採用された missions の score 合計」に置き換え、greedy と完全一致する
   評価軸の上で beam が並び替えをするか確認 (= 「beam が ordering で
   何を変えているか」のサニティチェック)。
3. **opponent mode を `true2p_light` に切替** — case3 G の知見を本実験に
   import。敵反応を folding した評価で、beam のオプション (a) 自軍 net_ships +
   (b) 敵反撃済みの差分を測れる。
4. **候補生成そのものの置換** — case3 result.md が指摘した第 1 番目の
   未踏領域。mission 列挙ロジック自体を変える (例: 守備優先の
   subset を必ず確保した上で攻撃 mission を beam) 。

優先順位は **2 → 3 → 1 → 4**。2 は 30 分で結論が出る最小コスト sanity check で、
beam の挙動異常か評価関数の不適合かを切り分ける。3 は case3 G の延長で
理論的根拠が強い。4 は本格的な再設計で次の plan.md レベル。

これらは本 result.md のスコープ外で、別 iter (`iter2_plan.md`) として切る
かどうかは user 判断。

## 採用済み memory への影響

- `project_om_finding` / `project_case5_validation` — Kaggle publicScore に
  頼らない方針は本実験でも維持 (ローカル 300戦のみで判断)
- `project_imitation_case1_phase3` — n<300 評価不可の教訓に従い 300戦実施

新規 memory 候補: 「case7 ベースで beam を被せると vs v4 で 32.3% に
劣化。heuristic score と horizon-end net_ships 評価関数の不一致が主因
として疑わしい」を `project_case8_beam_failure.md` で記録すべきか
(user 判断)。

## 再現手順

```bash
# Phase 1 (smoke 10 戦, ~30s)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 10 --seed 50000 --parallel 4 --no-save-replay

# 300戦合算 (本実験)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 150 --seed 50000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v8,baseline_v4 \
    --mode 1v1 -n 150 --seed 50500 --parallel 4 --no-save-replay

# あるいは wrapper スクリプト
uv run --directory bot python -m pipeline.rulebase.case8.evaluation.compare_v4

# tests
uv run --directory bot pytest tests/pipeline/rulebase/case8 -x
```

## 関連ファイル

- `bot/pipeline/rulebase/case8/baseline/planner/beam.py` — beam search core
- `bot/pipeline/rulebase/case8/baseline/planner/evaluator.py` — `score_commitments`
- `bot/pipeline/rulebase/case8/baseline/planner/candidate.py` — `commit_missions_in_order`
- `bot/pipeline/rulebase/case8/baseline/strategy.py` — `BEAM_ENABLED` 分岐
- `bot/pipeline/rulebase/case8/baseline/core/config.py:262-279` — `BEAM_*` constants
- `bot/tests/pipeline/rulebase/case8/` — 5 tests (smoke / beam_off=greedy / time_budget / evaluator unit)

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`, 実装コミット: 未 commit (本 result 執筆時点)
- 実行日時: 2026-05-04
