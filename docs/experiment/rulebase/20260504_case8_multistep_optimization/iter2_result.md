# Rulebase/case8 — Multi-step Beam (iter2 result)

> 作成日: 2026-05-05
> 対応 plan: [`iter2_plan.md`](./iter2_plan.md)
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — iter1 = 32.3% で却下
> - replay 分析 (iter1): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1230/`
> - replay 分析 (iter2): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1410_iter2/`
> - [`docs/experiment/rulebase/20260420_case3_rollout_ablation/result.md`](../20260420_case3_rollout_ablation/result.md) — score 補正系 5 連敗の予言

## 結論

**仮説は否定。iter2 も採用却下。** seat A 100戦のみ完走 (seat B は user 指示で中断)、合算 200戦は未取得だが、seat A の vs `baseline_v4` = **27.0%** はしきい値 ≥55% を大幅下回るため早期撤退。

iter1 (32.3%) → iter2 (27.0%) と **-3.7pp 悪化**。`true2p_sampled` (敵反応 folding を 5 ターンに 1 回) は計算予算には収まったが、勝率に貢献せず逆に不安定化を招いた。

## 数値

### 主要メトリクス: vs baseline_v4

| seat | n | v4 wins | v8 wins | **v8 win_rate** | v8 turn_p95 | timeouts |
|------|---|---|---|---|---|---|
| seat A (v4 first, seed 50000+) | 100 | 73 | 27 | **27.0%** | 0.569s | 0 |
| seat B (v8 first, seed 50500+) | — | — | — | — | — | — |
| 合算 | 100 (片 seat のみ) | 73 | 27 | 27.0% | 0.57s | 0 |

seat B は **user 指示により早期中断** (iter2 (A) 方針が飽和したと判断)。seat A 100戦のみで判定材料は十分 (iter1 比 -3.7pp、しきい値 ≥55% から -28pp)。

### 補助観察 (smoke 10 戦)

| 構成 | n | v8 win | v8 turn_p95 | 備考 |
|---|---|---|---|---|
| iter2 static + mission_score | 10 | 30% | 0.26s | greedy 同等 |
| iter2 true2p_light (毎 turn) + mission_score | 10 | 20% | 0.79s ⚠️ | 予算超 |
| iter2 true2p_sampled stride=5 + mission_score | 10 | 20% | 0.41s | 採用版、勝率改善せず |
| iter2 true2p_sampled (seed 51000+) | 10 | 40% | 0.49s | 別 seed 帯では 40%、seed variance 大 |

### しきい値判定

| 項目 | iter2_plan.md しきい値 | 実測 | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% (greedy 比 +5pp) | 27.0% (seat A のみ) | ❌ -28pp 大幅未達 |
| iter1 比改善 | 32.3% → 50% 復帰が最低線 | 27.0% (-3.7pp) | ❌ 後退 |
| seat 対称性 | ±10pp 未満 | seat A のみで判定不可 | — |
| turn_p95 | ≤0.7s | 0.569s | ✅ |
| timeouts | 0 件 | 0 件 | ✅ |

## 診断 — なぜ iter2 も失敗したか

### (1) `mission_score` 評価関数 + `static` で beam が greedy と等価

- `score_commitments(mission_score)` は採用 missions の `mission.score` 合計
- greedy ordering (score-desc) は **常に最大の合計を達成可能** (budget 制約が無い場合)
- 結果: beam が ordering を入れ替えても seed greedy より良い ordering が見つからない
- smoke `static + mission_score` 30% は iter1 の 32.3% と実質同等 → **改善余地なし**

### (2) `true2p_sampled` が **挙動を不安定化**

iter2 replay 分析 (`20260505_1410_iter2/result_{1,2}.md`) で確認:
- **t14 の自軍 ship 大量損失** (seed 51000: 22→2, seed 51001: 58→6) は iter1 と同じパターンが再現
- **planet thrash 連鎖が iter1 より悪化**:
  - seed 51000: planet#2 を turn 68-100 の 32 ターンに **5 回奪取・5 回喪失**
  - seed 51001: planet#21 を turn 76-92 の 16 ターンに **5 回奪取・5 回喪失**
- 主因推定: stride=5 の隙間 turn (敵反応 folding が無効) と発火 turn が混在 → **`ships_needed_to_capture` の判断が turn ごとに不一致** → 「奪取して即奪還される」mission を抑制できない

`true2p_light` (毎 turn) は計算予算 0.79s で破綻、`stride=5` で予算には収まるが上記の不安定化を招く。**両極のいずれも勝率改善せず。**

### (3) 速度最適化未達 (loop 指示の「性能を下げない条件で agent も速度最適化」)

- iter1 の同規模 (300戦) は ~10 分
- iter2 (200戦の seat A 100戦のみ) は **25分**、~5x の遅延
- 主因: `predict_enemy_reaction` が **WorldModel フル再構築** (`base_timeline` を全惑星で再計算)
- iter3 で改善余地あり (cache 流用、shallow copy)

### (4) base case7 の弱さは未対応

- iter1 で指摘済み (smoke で v4 vs v7 = 60-40)
- iter2 はすべて case7 base 上で実装、base 起因の劣化は織り込まれたまま

## 採用方針

- **iter2 は採用却下**
- iter1 と iter2 の連敗で、**「heuristic を beam で取り囲む方針」は構造的に飽和** が確定
  - case3 result.md (2026-04-20) の予言「score 補正系 5 連敗、次は MCTS / beam」を試したが、**beam search も同じ飽和の延長** であった
- production case4 (LB745) は引き続き現役
- `bot/src/dataset/selfplay/agents.py` の `baseline_v8` 登録は保持 (iter3 で改修するなら再利用)
- iter2 で導入した資産 (`planner/opponent_reaction.py`, `plan_moves_light`, `MovesPlan.accepted_missions`, `score_commitments_legacy`) は iter3 でも再利用可能

## iter3 で試す価値があるもの

iter1/iter2 の連敗から、**heuristic スコアの上に探索/評価関数を載せる方針は今後やらない**。次の方向は構造を変える 2 軸:

### 推奨: 候補生成側の改修 (case3 result.md の方針 (1))

mission builder 自体に **「直近 N ターン奪取済 / 奪還経験のある planet への mission 採用を抑制」** するフィルタを追加。replay 分析の planet thrash パターンに直接対応する設計。

具体案:
- `MovesPlan` または `WorldModel` に「直近 K ターンの planet 所有権変化履歴」を追加
- mission builder (capture/snipe/swarm) で「自軍が直近 5 ターン以内に 1 回以上奪われている planet」への mission を `mission.score *= 0.3` で減衰
- もしくは「同 planet への mission を直近 10 ターンで 2 回以上 commit している場合 skip」

これは beam search ではなく **既存 greedy の上で動く改修**。実装範囲は `agent.py` の状態追跡 + 1-2 個の mission builder。**1-2 時間で smoke + 200戦評価まで到達可能。**

### 副次: 速度最適化 + base 切替

| 案 | 効果 | コスト |
|---|---|---|
| `predict_enemy_reaction` の WorldModel 再構築を浅コピー化 | 200戦 25分 → 10分 | 1 時間 |
| base を case4 に切り替えて beam を被せる | case7 base 由来の handicap (~10pp) 切り分け | 30 分 |
| `BEAM_ENABLED=False` を default に戻し、case8 を「greedy + thrash filter」にする | shipped する場合は最低限 OFF が無難 | 5 分 |

## 採用済み memory への影響

新規 memory 候補: 「**heuristic を beam search で取り囲む方針は飽和** (iter1=32.3%, iter2=27.0%)。次の改善は候補生成自体の改修 (planet thrash filter) か学習ベース評価関数」を `project_case8_beam_saturation.md` で記録すべき (user 判断)。case3 result.md の予言を経験的に確認した結果でもある。

## 再現手順

```bash
# iter2 default (true2p_sampled stride=5 + mission_score)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 100 --seed 50000 --parallel 4 --no-save-replay

# replay 付き smoke (iter2 の挙動を replay 分析するなら)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 10 --seed 51000 --parallel 4

# tests
uv run --directory bot pytest tests/pipeline/rulebase/case8 -m "not slow" -x
```

## 関連ファイル

- `bot/pipeline/rulebase/case8/baseline/planner/opponent_reaction.py` — 敵反応 folding (case3/rollout.py から複製)
- `bot/pipeline/rulebase/case8/baseline/planner/evaluator.py` — `score_commitments` (mission_score) + `_legacy` (net_ships)
- `bot/pipeline/rulebase/case8/baseline/planner/beam.py` — `_select_evaluator` で mode dispatch
- `bot/pipeline/rulebase/case8/baseline/strategy.py` — `plan_moves_light` 追加、`true2p_sampled` 分岐
- `bot/pipeline/rulebase/case8/baseline/core/config.py:262-281` — `BEAM_*` constants (`BEAM_OPPONENT_MODE`, `BEAM_OPPONENT_SAMPLE_STRIDE`, `BEAM_EVALUATOR_MODE` 追加)
- `data/output/experiment/rulebase/case8/replay_analysis/20260505_1410_iter2/` — iter2 replay 分析

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
