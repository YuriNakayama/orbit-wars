# case7 「ルールベースに勝つ」ループ — iter05 RESULT

時刻: 2026-06-03 04:31 (cron tick 8)

## やったこと
BC warm-start + curriculum `noop(2) → python_v1` (**本物 baseline_v1**, host callback)
で直接学習。kl0.2, 12 iter 予定 (iter 6 で打ち切り)。

## Numbers (vs 本物 python_v1)
| iter | opp | win | reward | rollout |
|---|---|---|---|---|
| 0-1 | noop | 0.50 | ~0 | ~45s |
| 2 | python_v1 | **0.000** | -1.29 | 336s |
| 3 | python_v1 | **0.000** | -1.59 | 278s |
| 4 | python_v1 | **0.000** | -1.48 | 391s |
| 5 | python_v1 | **0.000** | -1.51 | 277s |
| 6 | python_v1 | **0.000** | -2.00 | 294s |

→ **本物 v1 に 1 勝もできず、reward は -2.0 へ悪化**。iter 6 で打ち切り
  (callback ~5min/iter で 0 勝、進展見込み薄)。

## ★決定的結論
**本物 baseline_v1 を相手に直接学習しても 0 勝** = memory `case6_pool_v1_rejected`
の「python_v1 は reward sparse で勾配消失」を **BC warm-start + curriculum 併用でも再現**。
モデルが 1 度も勝てない → PPO に正の学習信号が来ない → 学習が進まない。

これで打った手 (BCあり/なし × lite/full/self/**本物v1** × curriculum 1/3段 × anchor) が
**全て 0/10 (or 0勝)**。**case7 (この model 容量・特徴量・RL setup) では baseline_v1 に
勝てないことが確定的**。tuning や小規模 iter 追加では破れない構造的限界。

## 方針の再考 (重要)
「小規模学習 + 10戦で勝てる model」というゴールに対し、現アプローチ (case7 RL) は
**ローカル小規模では到達不能**と 8 tick で実証。残る現実的な道:

A. **GPU 大規模** (100-300 iter): memory で case1 が 300 iter/GPU で self-play 0.50 到達。
   ただし「self-play 0.50」≠「本物 v1 に勝つ」。case6 は GPU でも本物 v1 に 0/10 だった。
   → GPU を積んでも本物 v1 勝利は不確実。コストの割にリターン不透明。
B. **ゴールの再設定**: 「baseline_v1 に勝つ」は RL では到達困難と判明。
   - rulebase 側の改善 (case8 等、既に baseline_v1 と互角〜上) の方が確実。
   - または「baseline_v1 相当の rule を JAX で持ち、その上に薄い学習」= case_jax 路線。
C. **imitation の底上げ**: 出発点 (case9 模倣=0/10) が弱いのが根。より強い教師
   (baseline_v4/v8) の模倣に切り替えれば出発点が上がる可能性。

## NEXT ACTION (次 tick で判断)
- 現状を要約しユーザーに **方針判断を仰ぐ** (GPU 課金 vs ゴール再設定 vs imitation 底上げ)。
- 無断で GPU 課金や大方針転換はしない (重い一歩 + ゴール再設定はユーザー判断事項)。
- それまでループは「これ以上同じ小規模 RL を回しても 0/10」なので無駄打ちを避け待機。
