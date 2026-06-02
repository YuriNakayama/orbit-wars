# Reinforce/case5 — support_reward H6 time bonus + penalty (iter7) RESULT

> 関連: iter7_plan.md / hypotheses.md / iter3_result.md (H4 base 0.820) / iter6_result.md (H3 dense rejected)
> run_id: 20260531-122618__feature-support-reward__eb4092b__seed0
> commit: eb4092bc2354022bd0d1148ad5f71971492e8547
> case: reinforce_case5_kaggle_jax_train_h6_time_bonus
> 開始: 2026-05-31T12:26Z / 終了: 2026-05-31T16:03Z / コスト: ~$2.42 (RTX 4090 SECURE @ $0.69/h × 3.51h)

## Summary

H6 (時間方向報酬 — `time_bonus_coef=0.5` 勝利 terminal + `time_penalty_coef=0.001` 毎 turn) は **rejected**。
H4 (ratio coef=1.0, time bonus なし) base = lite last-10 **0.820** に対し H6 は **0.7625** (**-5.75pp**)。
学習自体は健全 (value_loss 0.0088, max approx_kl 0.0048) で training collapse は無いが、
時間圧力 = "速く勝て" のバイアスがリスクを取らせ最終勝率を下げた。
H3 (絶対 dense) の引き伸ばし副作用への対策として組んだが、**そもそも H4 では引き伸ばしが
問題化していなかった** ため、対策が単に勝率を下げた形。`time_bonus_coef=0.5` を
小さく (0.1) すれば改善余地はあるが、H6 = LAST hypothesis のため deepen は実施しない。

## Numbers (lite phase: opponent=baseline_jax_lite, iter6..200, n=195)

| metric | H6 (iter7) | H4 baseline (iter3) | Δ | note |
|---|---|---|---|---|
| last-10 win_rate (mean) | **0.7625** | 0.820 | **-5.75pp** | seed variance域(~±2pp) 外、明確な劣化 |
| first-10 win_rate (mean) | 0.178 | (n/a 直接比較不可) | — | lite phase 開始時、両者類似 |
| reward trend (last10 − first10) | +0.584 | +0.668 | -0.084 | 学習自体は機能、終盤伸び鈍化 |
| best_win_rate (全 iter) | 0.859 (iter4) | — | — | 早期 peak, 終盤やや低下 |
| value_loss last-10 mean | 0.00876 | 0.0066 | +0.002 | 健全 (H3 7163 のような破綻なし) |
| approx_kl last-10 mean | 0.00218 | 0.0024 | -0.0002 | 健全、target 0.02 余裕 |
| approx_kl max (lite) | 0.00482 | 0.005 | 同等 | clip 不要域 |
| iterations_run | 200 | 200 | — | 完走 |
| runtime | 3.51h | — | — | RTX 4090 SECURE |

## Diagnosis

**なぜ H4 (0.820) → H6 (0.7625) で -5.75pp?**

1. **時間圧力 = リスク選好バイアス**: `time_bonus = 0.5·(1 − turn/horizon)` は早期勝利に最大 +0.5
   ボーナスを付与。`time_penalty = 0.001·turn` は 500 turn 累積で -0.5。
   trajectoryへの圧力は **「速く勝て」** だが、case5 の対 baseline_jax_lite では
   惑星確保を急いだ攻め込みが過度にリスクを取り、引き分け/敗北側へ traj が流れた。
2. **bonus と PBRS の合計スケールが大きすぎる**: H4 base の shaping (`ratio Δ`) は
   ±0.05 程度の幅。それに加えて勝利時に +0.5 (terminal_reward = ±1 と同オーダー) を
   重ねたため、policy が PBRS シグナル < 時間圧力の優先順位を学習し、ratio 信号を弱めた。
3. **H3 と対をなす設計だが、H4 では「引き伸ばし」が問題化していなかった**:
   H6 は H3 (絶対 dense) の引き伸ばし副作用への対策として組んだ仮説。しかし
   H4 ratio (現行最良) では平均 game length が既に短く (~250 turn 程度)、
   引き伸ばしバイアスがないため対策が空振り → 副作用だけ残った。
4. **training collapse は起きていない**: value_loss / approx_kl は H4 base と同水準で
   学習構造は健全。「不適切な報酬設計に対し policy が正しく適応した結果、評価が下がった」
   という構図。

**例外条件適用**: hypotheses.md 例外条件「明確に劣化/引き伸ばし傾向が出た場合は rejected
として deepen しない」を適用。time_bonus_coef を 0.5 → 0.1 に下げる派生は理論的には
inconclusive 域に戻せる可能性があるが、本実験は support reward 設計軸の網羅であり、
H4 ratio coef=1.0 が最良で確定した時点で時間方向の探索は打ち切りが妥当。

## Decision

- **採否**: rejected (H4 比 -5.75pp、time bonus は PBRS と独立だが scale 過大で副作用)
- **次の一手**: なし。H6 は LAST pending hypothesis = list 消化完了。
  hypotheses.md 状態 を `completed` に遷移し loop 終了 → aggregate report。
  promote 候補は **H4 (iter3, run_id 20260528-004854, lite last-10 0.820)** のままだが
  promote/submit は OUT OF SCOPE (user 承認後に手動実行)。

## Artifacts

- model: `data/output/models/reinforce/case5_kaggle_jax_train_h6_time_bonus/runs/20260531-122618__feature-support-reward__eb4092b__seed0/best.pt`
- best ckpt: `best_i4_win0.8594.pt` (early peak)
- metrics: `data/output/models/reinforce/case5_kaggle_jax_train_h6_time_bonus/runs/20260531-122618__feature-support-reward__eb4092b__seed0/metrics.json`
- train log: `.../train.log`
- onstart log: `.../onstart.log`
