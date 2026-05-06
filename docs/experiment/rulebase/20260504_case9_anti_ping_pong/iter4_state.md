# iter4 — Loop Resume State

> 作成日: 2026-05-05
> Status: **iter4 評価実行中** (PID 86459, seed 5000-5199, 200戦)

## 進行中

- 変更: `MULTI_SOURCE_TOP_K: 5 → 8` + `THREE_SOURCE_PLAN_PENALTY: 0.75 → 0.85` (multi-source swarm 拡張)
- bypass=8 維持 (iter2 の知見)
- 評価: `compare_v4.py -n 100 -p 4 --seed 5000`
- ETA: ~100 分
- ログ: `/tmp/compare_v4_iter4.log`

## 完了したこと

- iter4_plan.md 作成
- config.py 2 行変更
- ruff/mypy green

## 次のループ周回でやること

1. PID 86459 重複ガード確認 → 進行中なら skip
2. 完了後:
   - `/tmp/compare_v4_iter4.log` 最終 summary を読む
   - iter4_result.md 作成 (iter2 比 +2pp で採択)
   - 採択 → iter5 で ACCUMULATE port 本格実装 (multi-source 強化を維持)
   - 棄却 → swarm 設定を元に戻し iter5 で別軸 (capture 強化 or ACCUMULATE 単独)

## 過去 iter の学び

- iter1 (cooldown 抑止): 46.0%, 雪崩崩壊シナリオ多発
- iter2 (bypass=8 + 値短縮): **49.5% (best)**, 雪崩は解消、僅差負けが残存
- iter3 (bypass=10 緩和): 47.8%/180戦中断, bypass 緩和は逆効果
- iter4 (multi-source 拡張): 仮説 = 「t=100 大型 launch 促進で僅差負けを勝ちに転換」
