# Rulebase/case11 — iter1 (PGS) Result: 0% Smoke, Implementation Issues

> 作成日: 2026-05-05
> 対応 plan: [`plan.md`](./plan.md) (3-iter sequential 検証 plan の iter1)
> 関連:
> - 関連 memory: `project_thrash_filter_harm.md` (heuristic 改修 8 連敗)

## 結論

**PGS v0 実装は完全敗北 (vs v4 = 0/10)、構造的な実装 issue 2 件が判明。** Stage A (30 戦) には進まず、**iter1 v0 を撤退** して issue を整理。次は v1 で fix するか、iter2 (NGS) に飛ぶか、構造ごと方針見直しを判断。

## 数値

### Smoke 結果

| 構成 | n | wins | win_rate | turn_p95 |
|---|---|---|---|---|
| v0 (PORTFOLIO_HORIZON=8) | 10 | 0/10 | **0.0%** | 0.042s |
| v1 (PORTFOLIO_HORIZON=20) | 10 | 0/10 | **0.0%** | 0.058s |
| v1 + replay 5戦 | 5 | 0/5 | 0.0% | 0.033s |
| v2 (script_harass send //4→//2 + capture_safe send=need+4) | 10 | **0/10** | **0.0%** | 0.038s |
| **v3 (capture_aggressive で reserve=max(5, prod*3) 確保)** | 10 | **0/10** | **0.0%** | 0.058s, avg_turns 220 |

turn_p95 0.03-0.06s = case4 (~0.65s) の 1/10。**PGS が出している move が極端に少ない**。**4 連敗で PGS の構造問題** が確定。

## 診断

### Issue 1: PORTFOLIO_HORIZON=8 では capture が評価されない (v0 → v1 で fix)

- typical capture ETA = 10-12 turn (近距離 neutral でも)
- `simulate_planet_timeline(horizon=8)` は eta=11 の arrival を見れない
- → `evaluate_assignment(idle) == evaluate_assignment(capture_safe)` 同点
- → hill climb は idle から動かず agent は何もしない

**Fix**: `PORTFOLIO_PLAYOUT_HORIZON: 8 → 20` で v1 へ。score は differentiate するが win_rate は変わらず。

### Issue 2: Evaluator が sent ships を src から差し引いていない (v1 でも未 fix)

`evaluate_assignment` は each planet を独立に `simulate_planet_timeline` するが、 **送出 ships を src から減らす操作が無い**。結果:

- `send 2 ships to enemy` と `send 9 ships to enemy` で **score 同等** (neutral capture 成功率は same、src 残艦は両方とも同じ "starting ships + production")
- → PGS は **小さい move を好む** (低リスクに見える)
- 実際は src は ships を失うので、large capture の方が src 内残量への penalty が出るべき

**Replay 上の挙動 (seed 90100, t46)**:
```
| 50 | self | enemy_planet_attack | sent 2 ships at enemy planets |
| 52 | self | enemy_planet_attack | sent 2 ships at enemy planets |
| 54 | self | enemy_planet_attack | sent 2 ships at enemy planets |
```
2 ships の連射 (= `script_harass` の `src.ships // 4 = 2`) を選び続け、capture できず thrash。

### Issue 3: Early-game expansion が追いつかない (構造問題)

t=20 まで self vs opp が互角だが、t=20→30 で opp が +2 planets / +55 ships、self は +0 planets / -8 ships で逆転される。

PGS の評価関数が **「home を空にする capture_aggressive」を penalty 視** するため、初期に必要な「全力で neutral 確保」が行われない。case4 base の greedy は「あるなら全力で取りに行く」が default で、そっちが正解だった。

## 採用方針

- **iter1 v0/v1 は採用却下** (PGS の現実装は production case4 に対し 0%)
- 実装は残置 (`PORTFOLIO_ENABLED=False` で case4 等価動作する事を unit test で保証)
- **case11 default を `PORTFOLIO_ENABLED=False` に戻す** (本 result 執筆完了直後に修正)

## 次の選択肢

### (A) iter1 v2: Evaluator fix (推奨)

Issue 2 を fix:
1. `evaluate_assignment` で **sent_per_src を計算**、各 src の simulate_planet_timeline 開始 ships を `src.ships - sum(sent_from_src)` に
2. `script_harass` の発射量を `// 4` → `// 2` に増やす
3. `script_capture_safe` の send_cap を `need + safety_buffer` に強化

期待: large move の score が増し、PGS が `capture_aggressive` 系を選ぶ。30戦で 30-50% 期待。

コスト: 30 分実装 + 30 分テスト + 30 分 30 戦 ≈ 1.5 時間

### (B) iter2 (NGS) に飛ぶ

NGS は PGS の上位互換 (enemy reaction folding)、しかし PGS が機能していない以上 **NGS で改善する確率は低い**。本提案は推奨しない。

### (C) iter3 (NaïveMCTS) に飛ぶ

action space を sampling-based MCTS で広範囲 search。実装コスト最大 (3-4日)、ただし PGS の構造問題 (script-only 探索) を bypass できる可能性。最終手段。

### (D) Portfolio search 系を全撤退、別方向

heuristic 系 + portfolio 系合わせて **9 連敗**確定。次は学習評価関数 / 別 agent family。本ディレクトリ撤退。

## 推奨

**(A) iter1 v2 → 30戦で再評価**。PGS の「smaller is better」bug を fix できれば本来の portfolio search の力を測れる。これで <40% なら (D) 撤退、≥50% なら iter2 (NGS) で追加 fold-up を試す。

## 関連ファイル

- `bot/pipeline/rulebase/case11/baseline/core/config.py:PORTFOLIO_*` — config 4 個 (HORIZON 8→20 変更済)
- `bot/pipeline/rulebase/case11/baseline/planner/scripts.py` — 7 script
- `bot/pipeline/rulebase/case11/baseline/planner/evaluator.py` — Issue 2 (sent ships 未控除) 含む
- `bot/pipeline/rulebase/case11/baseline/planner/portfolio_greedy.py` — hill climbing
- `bot/tests/pipeline/rulebase/case11/` — 4 unit tests pass
