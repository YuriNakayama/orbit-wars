# iter10 — Loop Resume State

> 作成日: 2026-05-06
> Status: **iter10 未開始**、case9 = iter2 設計 + iter6 cache (iter8 の HARASS=3 は revert)

## iter1-9 確定結論

case9 設計の **真値 ~50%** (case4 同等)。+5pp 達成は seed 偶発レベル。
- iter1-7: 全棄却 (cooldown 系小修正)
- iter6: plan_shot cache 採用 (基盤、30% 高速化)
- iter8: HARASS=3 で 200戦 55.5% 採択も、iter9 300戦 confirm で 48.3% に回帰
- iter1-9 の 200/300 戦結果総合で **+5pp 不可確定**

## iter10 候補 (優先順)

### A: Loop 終了 (推薦)

9 iter (10+ 時間) を投下し +5pp 達成は seed 偶発のみ。完了条件 60% は遥か遠い。
case9 anti-ping-pong は **iter2 設計を最終形態として確定**、新規 case で別アプローチ。
- memory に case9 失敗事例 (cooldown 系では届かない、ACCUMULATE 単独 port 失敗) を記録
- CronDelete 11c931e9 で cron 停止
- ユーザー判断仰ぎ

### B: case7 ACCUMULATE + STAY_BURST 同時 port (大型 iter)

iter5 ACCUMULATE 単独 port は -7pp で失敗。STAY_BURST も同時に配線すれば
case7 同等の挙動になり、case9 でも効く可能性。実装は ~1000 行追加 + 5+ 周回。
**勝てる保証なし**、優先度低。

### C: case10 で完全に別アプローチ

- case4 大改造、capture mission の score weight 強化
- 完全に別の戦術 (kamikaze、sniper 多用)
- case9 はアンチパターン保管としてキープ

## 推薦判断: A (Loop 終了)

理由:
- 9 iter の投下で +5pp が seed 偶発のみと確定
- case9 設計空間内 (anti-ping-pong + 既存 mission の閾値) では真値 50% から動かない
- 完了条件 60% は **抜本的な戦術設計変更 (case10) でしか到達不可**
- cron loop の小さな iter 続行は探索コストの無駄

## 次の周回でやること

1. **memory 候補書込み**: docs/MEMORY/ に「case9 anti-ping-pong: 9 iter で +5pp 不可、真値 ~50%、iter6 plan_shot cache のみ採用、HARASS=3 は seed 偶発」記録
2. **CronDelete 11c931e9** で cron 停止
3. ユーザー完了報告 (次は case10 or 別タスク判断)

## 過去 iter サマリ

- iter1: 46.0% / iter2: 49.5% (best 設計) / iter3: 47.8% (中断)
- iter4: 47.0% / iter5: 42.5% (ACCUMULATE) / iter6: 49.0% (cache 採用)
- iter7: 52.0% (300戦) / iter8: 55.5% (200戦、seed 偶発) / **iter9: 48.3% (300戦、確定)**
