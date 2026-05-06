# iter8 — Loop Resume State (RUNNING)

> 作成日: 2026-05-06
> Status: **iter8 評価実行中** (PID 93911, seed 10000-10199, 200戦)

## 進行中

- 変更: `HARASS_MIN_TARGET_PRODUCTION: 2 → 3` (1 行変更)
- 仮説: 低 production target への harass 発火を抑制し、余剰 ship を capture/reinforce
  に回す → 勝率改善
- 評価: `compare_v4.py -n 100 -p 4 --seed 10000`
- ETA: ~70 min (iter6 cache 経由)
- ログ: `/tmp/compare_v4_iter8.log`

## 採否しきい値

- **≥55%**: 採択 (+5pp 達成)
- **51.5%-55%**: 弱採択、iter9 で 300戦 confirm
- **<51.5%**: 棄却、iter9 で別軸 (capture score weight、ACCUMULATE+STAY 同時 port)

## 次のループ周回でやること

1. PID 93911 重複ガード確認
2. 完了済みなら結果分岐:
   - 55% 以上 → 採択コミット、iter9 で 300戦 confirm
   - 51.5-55% → 弱採択、iter9 で 300戦 confirm
   - 51.5% 未満 → 棄却 + HARASS_MIN_TARGET_PRODUCTION=2 に戻す + iter9 別軸

## 過去 iter サマリ

- iter1: 46.0% (cooldown 抑止)
- iter2: 49.5% (best 設計、200戦)
- iter3: 47.8% (中断、bypass=10)
- iter4: 47.0% (multi-source)
- iter5: 42.5% (ACCUMULATE port、最低)
- iter6: 49.0% (plan_shot cache、採用)
- iter7: 52.0% (300戦 confirm、+5pp 未達)
- **iter8: HARASS 発火閾値 2→3、評価中**
