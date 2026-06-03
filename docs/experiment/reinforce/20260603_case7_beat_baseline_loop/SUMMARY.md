# case7 「v1 に勝つ」ループ — 総括 (25 tick 時点)

時刻: 2026-06-03 17:01 (cron tick 25)

## ゴールと結論
- ゴール: 小規模学習で baseline_v1 (rulebase) にローカル対戦で勝てる model。
- 結論: **小規模ローカル RL では baseline_v1 に勝てない (0/10) が確定**。ただし
  「弱い〜互角の学習相手 (rl_v0) には勝てる (1.00)」model は作れた。v1 越えは scale 必須。

## ★主要な成果 (4 commit)
1. **horizon=200 terminal 報酬消失バグの発見・修正** (c68→horizon500)。
   ゲームが ~497 turn 続くのに horizon=200 で打ち切り → 勝敗報酬が毎 step 0 →
   shaping ノイズだけで学習。13 tick の 0/10 の主因。修正で学習信号復活。
2. **resume_from + incremental metrics** (best.pt から追加学習、iter ごと metrics 追記)。
3. **memory features** (実 launch history を学習 rollout に記録、train/eval gap 縮小)。
4. **per-iter checkpoint + ckpt-sweep model 選択** (self-play win は外部汎化と無相関、
   best.pt=最後 では peak を逃すため)。

## ★確立した知見 (memory 化済)
- `project_reinforce_horizon_terminal_reward_bug`: horizon は必ず 500。
- `project_reinforce_unbeatable_opponent_harmful`: 勝てない強相手 (lite/v1) を学習相手に
  すると reward 飽和で劣化。self-play 回しすぎも self distribution に overfit して劣化。
  → 学習相手は self_snapshot pool で非飽和に保ち、model は外部 eval で選ぶ。
- pure self-play は `priority: uniform` + `late_full_prob: 0` (f_hard は full 強制混入)。

## 到達点 (model 強さ)
| 相手 | 到達勝率 | 種別 |
|---|---|---|
| random / il_v0 / self_snapshot | ~0.85-1.0 | ✅ 勝てる |
| **rl_v0 (少し弱い学習モデル)** | **1.00** (iter12) | ✅ 達成 |
| baseline_jax_lite (v1相当) | ~0.06-0.25 plateau | ❌ 天井 |
| baseline_v1 (本物) | 0/10 | ❌ 天井 |

ベスト model = **iter12** (`local_20260603T031704Z/best.pt`, self-play ladder 16 iter)。

## なぜ v1 に勝てないか (research 裏づけ)
- 敗因は production gap (score 51:16000 級、v1 の拡張・生産に完敗)。
- Generals.io 論文 (arXiv 2507.06825): 同レシピ (BC+self-play+PBRS+memory) で領土拡張
  ゲーム攻略に **H100×36h**。本ループは CPU 数分 = scale が 1000倍以上不足。
- JAX-native な「学習可能な外部相手」が lite/full しか無く、両方とも飽和気味。

## 探索は飽和
- BC / curriculum / shaping (ratio/combined/ratio_prod) / anchor / self-play 長短 /
  memory features / 本物 v1 直接 / production shaping — 全試行で v1 は 0/10。
- iter12 を起点にした追加学習はどの方向も劣化 (lite=飽和害、self続行=overfit害)。
- **小規模 RL レシピ探索は飽和**。残るは scale (GPU) か別 family (rulebase)。

## 残る選択肢 (ユーザー判断)
1. GPU scale (RunPod、~$1.5+、research 準拠だが memory: GPU でも v1 0/10 実績あり)。
2. 別 family: rulebase/case8 は既に v1 互角〜上 (確実)。
3. 現状で一区切り (学習基盤健全化 + 弱相手攻略 + 4 commit は達成済)。
