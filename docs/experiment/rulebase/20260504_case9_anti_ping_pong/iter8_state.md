# iter8 — Loop Resume State

> 作成日: 2026-05-06
> Status: **iter8 未開始**、case9 = iter2 設計 + iter6 cache (final 状態)

## iter1-7 全体結論

case9 anti-ping-pong は **iter2 設計が真値 50-52%、+5pp 達成は不可**。
- ping-pong 件数 -11.5%、launches -17.7% で抑制機構は確認
- ただし vs baseline_v4 勝率は case4 同等、case4 を超える戦術差は出せず

## iter8 候補 (優先順)

### A: Loop 終了判断 (推薦)

7 iter にわたって case9 を探索したが、
- iter1-7 で +5pp 達成しなかった
- iter2 設計を 300戦 confirm 済 (52.0%、CI 上限 57.5% で borderline 触る)
- 完了条件 (60%) は遥か遠い

**残作業**: case9 を current best (iter2 設計 + iter6 cache) で確定し、
docs/MEMORY に case9 の知見を残して loop 停止 (CronDelete 11c931e9)。
case9 を超える次の挑戦は別 case (case10) で別アプローチ。

### B: case7 ACCUMULATE + STAY_BURST 同時 port (大型 iter)

iter5 ACCUMULATE 単独 port は -7pp で大失敗。STAY_BURST も同時に配線すれば
case7 同等の挙動になり、case9 でも効く可能性。実装は ~1000 行追加 + 5 周回。
**勝てる保証なし**、優先度低。

### C: case10 で別アプローチ

case4 大改造、capture mission の score 強化、または完全に別の戦術 (kamikaze、
sniper 多用) を case10 で。case9 は keep (アンチパターン保管)。

## 推薦判断: A (Loop 終了)

理由:
- 30分 cron × 7 iter = 3.5 時間以上を投下、+5pp 未達
- iter6 plan_shot cache の高速化は副次採用
- ACCUMULATE 単独失敗で case7 系 port の信頼性が低下、フル port するなら
  **新 case を切るのが安全**
- 評価コストが大きいため、低価値 iter を続けるより loop を一旦止めて
  ユーザー判断を仰ぐ方が合理的

## 次のループ周回でやること

1. **memory 候補書込み**: docs/MEMORY/ に「case9 anti-ping-pong: iter2 設計が
   best (300戦 52.0%)、+5pp 達成不可、ACCUMULATE 単独 port は -7pp」記録
2. **CronDelete 11c931e9** で cron 停止
3. ユーザーに完了報告 + 次の方針確認 (case10 別アプローチ or 別タスク)

## 過去 iter サマリ

- iter1: 46.0% (cooldown 抑止)
- iter2: **49.5% (best 設計、200戦)**
- iter3: 47.8% (中断)
- iter4: 47.0% (multi-source)
- iter5: 42.5% (ACCUMULATE)
- iter6: 49.0% (plan_shot cache、採用)
- **iter7: 52.0% (300戦 confirm、+5pp 未達確定)**
