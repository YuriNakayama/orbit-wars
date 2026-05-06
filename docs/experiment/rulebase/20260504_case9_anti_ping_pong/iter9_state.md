# iter9 — Loop Resume State

> 作成日: 2026-05-06
> Status: **iter9 未開始**、case9 は iter8 採択状態 (HARASS_MIN_TARGET_PRODUCTION=3)

## 直近の状態 (iter8 締め)

- **iter8 採択 (200戦 55.5%、+5pp ボーダー突破)**
- iter1-8 で初めての採択
- 設定: `HARASS_MIN_TARGET_PRODUCTION: 2 → 3` の 1 行変更が決定打

## iter9 でやること (300戦 confirm)

case9 を変更せず、現在の設計 (iter2 + iter6 cache + iter8 HARASS=3) のまま 300戦再評価:

1. PID/重複ガード確認 (no compare_v4)
2. `compare_v4.py -n 150 -p 4 --seed 11000` (300戦) を起動
3. ETA: ~100 min

## 採否分岐

- **300戦 ≥60%**: **完了条件達成** → docs/MEMORY に成功事例書込み + CronDelete 11c931e9
- **300戦 ≥55%**: 採択は維持、iter10 で更にチューニング
- **300戦 <55%**: 200戦の seed 偶発と判定、iter10 で別軸

## 過去 iter サマリ

- iter1-7: 全棄却 (best は iter2 49.5%)
- iter6: plan_shot cache 採用 (30% 高速化)
- **iter8: HARASS=3 で 55.5% (200戦) 採択!**
- **iter9: 300戦 confirm で完了条件 60% 判定**
