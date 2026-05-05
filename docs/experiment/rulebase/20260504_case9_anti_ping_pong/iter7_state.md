# iter7 — Loop Resume State

> 作成日: 2026-05-06
> Status: **iter7 未開始**、case9 = iter2 設計 + iter6 plan_shot cache (高速化基盤)

## 直近の状態

- iter6 plan_shot cache: 採用 (49.0% vs iter2 49.5%、-0.5pp 許容範囲)
- 評価所要時間: 100 min → 70 min (30% 高速化)
- best 勝率: **iter2 = 49.5% (200戦)** ← 依然として best、+5pp 未達

## iter7 候補 (優先順)

### A: cooldown 値の微調整 (iter2 ベース)

iter1 (PAIR=3, HARASS=5) → iter2 (PAIR=1, HARASS=2) で +3.5pp 改善。中間値を試す:
- `PING_PONG_PAIR_COOLDOWN_TURNS: 1 → 2` (1 行変更)
- 200戦評価で iter2 比 +2pp 以上なら採用
- リスク: iter1-2 の知見からは「短いほど良い」傾向、+2 で逆効果の可能性

### B: REINFORCE_MIN_DEFICIT 引き上げ

- iter1 で=3、iter2 で=1。中間=2 を試す
- 小規模脅威への過剰反応をやや抑制、雪崩崩壊は防げる程度
- 1 行変更で評価可能

### C: 300戦再評価で iter2 best を確定

- 70 min × 1.5 = ~100 min で 300戦
- Wilson CI が ±7pp → ±5pp に縮小
- iter2 が +5pp 達成しているかの **真値検証**
- 完了条件達成判定の最終的な根拠になる

### D: build_world の追加最適化 (iter6 路線継続)

- `WorldModel.__init__` で `base_timeline` 構築が重い (timeline simulation)
- 今 turn の self planet についてのみ計算するよう絞れば 50% 短縮見込み
- ablation 必須

## 推薦判断

**候補 C (300戦再評価)** が最も価値高:
- iter1-6 の累積で iter2 が best、これを confirmed best にできれば「+5pp 完了条件」の達成可能性を確定的に判断できる
- もし 300戦で 55% 以上達成していれば **loop 完了**
- 未達なら +5pp は cooldown tuning 系では届かないと結論し、別軸 (ACCUMULATE+STAY 同時 port、case4 ベース大改造) に振り切る

候補 A/B は 300戦結果次第で iter8 に積む。

## 次のループ周回でやること

1. case9 = iter2 等価 (ACCUMULATE_ENABLED=False) を確認
2. `compare_v4.py -n 150 -p 4 --seed 9000` (各 seat 150戦 = 300戦) を起動
3. ETA ~100 分 (iter6 の 70 min × 1.5)

## 過去 iter サマリ

- iter1 (cooldown 抑止): 46.0%
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)**
- iter3 (bypass=10): 47.8%/180中断
- iter4 (multi-source): 47.0%
- iter5 (ACCUMULATE port): 42.5%
- **iter6 (plan_shot cache)**: 49.0%、ablation 採用、30% 高速化

## 既知の todo

- snapshot_update.py の出力先バグ (worktree root に tests/ 作成) は別 commit で修正
