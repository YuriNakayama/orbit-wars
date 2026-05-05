# case10 iter1 — Loop Resume State (RUNNING)

> 作成日: 2026-05-06
> Status: **iter1 評価実行中** (PID 53445, seed 12000-12199, 200戦)

## 進行中

- case10 = case4 + 5 定数変更 (capture 強化 + sniper/kamikaze 多用):
  - `STATIC_NEUTRAL_VALUE_MULT: 1.4 → 1.6`
  - `EARLY_NEUTRAL_VALUE_MULT: 1.2 → 1.4`
  - `SNIPE_VALUE_MULT: 1.12 → 1.30`
  - `HARASS_MIN_SRC_RESERVE: 10 → 6`
  - `HARASS_PRODUCTION_STEAL_TURNS: 5 → 8`
- 評価: `compare_v4.py -n 100 -p 4 --seed 12000`
- ETA: ~100 分 (case10 は plan_shot cache 未移植のため case9 iter6 比で遅い)
- ログ: `/tmp/compare_v4_case10_iter1.log`
- PID: 53445

## 採否しきい値

- ≥55%: 採択候補、iter2 で 300戦 confirm
- 51-55%: 弱採択、iter2 で 300戦 confirm
- <51%: 棄却、iter2 で別軸 (5 定数の ablation)

## 次の周回でやること

1. PID 53445 重複ガード確認
2. 完了後 iter1_result.md 作成 + 採否分岐

## memory との関係

- `project_heuristic_search_saturation`: case8/9/10/11/12 で heuristic 53% 飽和 → 本 iter1 はそれを超えるかチャレンジ
- `project_case9_anti_ping_pong_2026_05_06`: 9 iter 失敗の教訓 (cooldown 系小修正 NG) を踏まえ、別軸 (capture/snipe weight) を試行

## 既知の todo

- plan_shot cache を case10 にも移植 (iter6 の知見、別 commit ablation 候補)
- snapshot test の `test_agent_runs_1v1_to_done` がフラッキー (case9 と同パターン)
