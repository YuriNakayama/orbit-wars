# case7 ループ — iter14 PLAN: self-play ratchet (PFSP scaling)

時刻: 2026-06-03 14:31 (cron tick 20)

## 前提 (確定)
- iter12 (self-play ladder) = rl_v0 0.90 がベスト model。
- iter13 (lite を学習相手に追加) = 有害、rl_v0 0.90→0.40 劣化
  (memory `project_reinforce_unbeatable_opponent_harmful`)。
- 教訓: 学習相手は非飽和 (self_snapshot pool) に保つ。強相手は scale で徐々に。

## iter14 方針: PFSP ratchet
iter12 BEST から resume、**純 self-play pool で 20 iter** (snapshot_every=2, cap6, f_hard)。
過去の自分が学習とともに強くなる → 「少し前の自分」を倒し続けて漸進的に強化。
これが PFSP の本来の使い方 = 一足飛びに強相手を入れず、self で ratchet。

## 期待 / 判定
- self-play win が安定して >0.5 を保てば学習継続 (PFSP 健全)。
- 完走 → 10戦 vs rl_v0。**0.90 を超える / 維持できるか** = self-play scaling の効果。
- おまけで vs lite を probe: 非飽和度 (0.25→?) が上がれば「梯子の次段が近づく」サイン。

## 留意
- best.pt は win_rate>=best で self の 1.0 に張り付きがち → iter12 と大差ない可能性。
  本実験は「self-play を長く回すと強くなり続けるか」の検証。
- v1 直接はまだ天井。本 iter は iter12 を起点に self scaling の伸びしろを見る段階。
- ローカル CPU なので 20 iter ~40min。GPU 不使用 (小規模方針)。
