# case7 「ルールベースに勝つ」ループ — iter03 PLAN

時刻: 2026-06-03 02:37 (cron tick 3 続き)

## ここまでの確定事実
- 3 model 全て vs baseline_v1 = **0/10** (16-iter / 生BC / BC-RL 14iter)。
- 出発点 (BC=case9 模倣) が既に 0/10。小規模 CPU RL では動かない。

## web search で得た知見 (loop 指示に基づき調査)
"PPO vs scripted bot / sparse reward / curriculum / self-play / small compute":
- **dense→sparse の段階的 curriculum が定石**: 簡単で dense reward なタスクから始め、
  徐々に難しく sparse な相手へ。いきなり強い相手は学習信号が出ない。
- reward shaping は sparse 穴埋めに有効 (PBRS ratio は採用済)。
- self-play は「自分の版」と戦い漸進的に強くする (pool は実装済)。
- 出典:
  - [Self-Adaptive Reward Shaping (arXiv 2408.03029)](https://arxiv.org/pdf/2408.03029)
  - [Shaping Sparse Rewards: Semi-supervised (arXiv 2501.19128)](https://arxiv.org/html/2501.19128v1)
  - [PPO + Curriculum (Medium)](https://medium.com/@adityabanerjee171/reinforcement-learning-training-a-ppo-agent-to-play-ping-pong-with-curriculum-learning-1693c2370f97)
  - [Self-play essence (Medium)](https://medium.com/@kaige.yang0110/the-essence-of-selfplay-in-reinforcement-learning-and-muzero-ef5d304a5584)

## 診断: なぜ 0/10 のままか
- 現 curriculum は noop(2) → baseline_jax_lite と **段差が大きすぎ** (lite は v1 相当で強い)。
  研究の「dense→難へ段階的に」に反する。
- BC anchor (kl_beta 0.1) は弱く、bc_kl 0.23 まで drift = 勝てない方向へ動いている恐れ。

## iter03 でやること (次 tick)
1. **3 段 curriculum + 強め anchor**: `noop → self_snapshot(自分) → baseline_jax_lite`。
   中間に「勝てる相手 (自分の過去)」を挟み勝ち経験を確保。kl_beta 0.1→0.3。
   ratio shaping 維持。BC warm-start から開始。
2. iters を稼ぐため self_snapshot 中心 (lite より軽い) にして ep/horizon 据え置き。
3. 完走 → 10戦 vs baseline_v1。0/10 から動くか。
4. それでも 0/10 なら **compute scale が主因**と判断 → 認証不要方針なので
   RunPod GPU で iter 数を一桁増やす (200 iter 級) に切替検討。

## 留意
- best.pt は **main repo 絶対パス**で参照 (worktree data symlink は不安定)。
- 10戦は方向性確認のみ (大規模検証は避ける方針)。動いたら 30-50 戦で再確認。
