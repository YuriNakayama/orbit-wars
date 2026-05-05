# rulebase/case10 — capture_kamikaze (iter2 plan, conditional on iter1)

> 作成日: 2026-05-06
> 関連: `iter1_plan.md`, `iter1_state.md`
> Status: **iter1 評価完了待ち** (PID 53445, ETA ~100min)

## iter2 仮説の分岐 (iter1 結果次第)

iter1 は 5 定数を同時に変更した「複合介入」。結果次第で iter2 方向が変わる:

### iter1 採択 (≥55%) の場合 → **300戦 confirm**

設定変更なし。`compare_v4.py -n 150 -p 4 --seed 13000` で 300戦評価。
完了条件 (≥60%) 達成判定 + Wilson CI 確認。

### iter1 弱採択 (51-55%) の場合 → **どの定数が効いたか ablation**

5 定数を 1 つずつ default に戻して各々 200戦評価:
- A: `STATIC_NEUTRAL_VALUE_MULT 1.6 → 1.4` (capture 強化を解除)
- B: `EARLY_NEUTRAL_VALUE_MULT 1.4 → 1.2`
- C: `SNIPE_VALUE_MULT 1.30 → 1.12`
- D: `HARASS_MIN_SRC_RESERVE 6 → 10`
- E: `HARASS_PRODUCTION_STEAL_TURNS 8 → 5`

**最も悪化した変更が「効いた定数」**。コスト: 200戦 × 5 = 約 8 時間。
非現実的なので **iter2 では A と D だけ ablation** (capture 系と kamikaze 系の代表):
- iter2: A 解除 → 残 4 定数効果を見る
- iter3: D 解除 → 残 4 定数効果を見る
- それぞれ 200戦、合計 ~3.5 時間

### iter1 棄却 (<51%) の場合 → **逆方向 (defense 寄り) を試す**

Capture 弱化 / kamikaze 抑制方向で 1 セット試行:
- `STATIC_NEUTRAL_VALUE_MULT 1.6 → 1.2` (default 1.4 より低く)
- `HARASS_MIN_SRC_RESERVE 6 → 14` (kamikaze 抑制)

これで勝率改善するなら **iter1 の方向が逆効果** と確定。改善しないなら
heuristic 系探索の 53% 壁が再確認される。

## 共通の方針

- 200戦 default、採択しきい値 +5pp (≥55%)、弱採択時のみ 300戦 confirm
- 5 iter 連続棄却で loop 終了 (cron prompt のガード規定)
- memory `project_heuristic_search_saturation` の 11 連敗パターンと類似の場合は 3 連続棄却で早期終了

## 次の cron fire でやること

1. PID 53445 重複ガード確認
2. iter1 完了済 → このファイルの分岐に従って iter2 を起動
3. 進行中 → no-op
