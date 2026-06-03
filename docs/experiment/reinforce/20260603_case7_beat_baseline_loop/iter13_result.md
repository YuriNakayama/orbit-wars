# case7 ループ — iter13 RESULT: lite resume は plateau

時刻: 2026-06-03 14:01 (cron tick 19)

## やったこと
iter12 (ladder で rl_v0 に 0.90) の best から resume、self_snapshot(2) → baseline_jax_lite
で 14 iter (horizon=500, ratio/1.0)。lite が probe で 0.25 と非飽和だったのを受け、
「梯子の次の段 (lite=v1相当)」を登れるか検証。

## lite-stage の win 推移 (iter 2-10)
```
0.00, 0.00, 0.25, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00
```
reward: -1.95, -1.91, -1.19, -2.0, -2.0, -1.12, -2.0, -2.0, -2.0

## ★結論: plateau (登れず)
- 序盤の -1.95 → -1.19 は **学習ではなくノイズ** (その後 -2.0 に戻る)。
- lite vs win は **平均 ~0.06 で横ばい**、上昇トレンドなし。
- ladder で model を強化し lite で時々勝てる (0.25) ようになったが、
  **1/4 勝の勾配では lite を一貫して攻略できず plateau**。
- = lite (v1相当) は小規模 RL の硬い天井。「弱い相手 (rl_v0) には勝てるが、
  v1級には届かない」境界がここにある。

## ループ全体の到達点 (確定)
| 相手 | 到達勝率 | 種別 |
|---|---|---|
| random / il_v0 / self_snapshot | ~0.85-1.0 | ✅ 勝てる |
| **rl_v0 (少し弱い学習モデル)** | **0.90** (iter12) | ✅ **達成** |
| baseline_jax_lite (v1相当) | ~0.06 plateau | ❌ 天井 |
| baseline_v1 (本物) | 0/10 | ❌ 天井 |

## 切り分け済の事実
1. horizon=200 バグ (修正済) が 13 tick の 0/10 主因だった。
2. 修正後、self-play / rl_v0 では学習が機能し勝てる (0.90)。
3. lite/v1 は reward 飽和気味で小規模 RL の天井。research (Generals.io) 通り
   ここを越えるには scale (H100×36h 級) が必要。

## NEXT
- iter13 完走 → 10戦 vs v1 (確認のみ、plateau から 0/10 予想)。
- 小規模で v1 級を攻略するのは構造的に困難と再確認。
  loop の本来価値 = 「学習基盤の健全化 + 弱い相手攻略の実証 + horizon バグ発見」は達成済。
- これ以上の小規模 lite/v1 攻略は ROI 低い。scale (GPU) かゴール再設定はユーザー判断。

## ★iter13 最終測定 (14:14) — lite 学習は有害
| 相手 | iter13 後 | iter12 (self-play only) |
|---|---|---|
| baseline_v1 | 0/10 | 0/10 |
| **rl_v0** | **4/10 (0.40)** | **9/10 (0.90)** ← 良い |

- vs v1 は予想通り 0/10 (plateau から transfer せず)。
- ★**rl_v0 への勝率が 0.90 → 0.40 に劣化**: lite-stage の saturated -2.0 が方策を
  悪い方向へ押し、best.pt も lite-stage の win 0.75 で保存 (iter12 の純 self-play
  model より rl_v0 で弱い) → **強すぎる相手で学習すると、勝てていた相手にも負け始める**。

## ★確定した教訓 (重要)
- **「勝てない相手 (lite/v1) で学習する」のは有害**: reward 飽和で勾配が壊れ、
  既に勝てる相手 (rl_v0) の性能まで破壊する。
- **正解は self-play (勝てる相手) で reward 非飽和を保つこと** = iter12 のレシピ。
- iter12 model (rl_v0 0.90) が現状ベスト。lite/v1 を学習相手に加えるのは逆効果。
- → 小規模で v1 に勝つのは「強すぎて学習に使えない」というジレンマ。scale で
  徐々に強い self snapshot を積むしかない (Generals.io 路線)。

## ベスト model
- **iter12** (`local_20260603T031704Z/best.pt`): self-play ladder のみ、rl_v0 0.90。
  以後の resume はこれを起点にすべき (iter13 の lite-degraded model は使わない)。
