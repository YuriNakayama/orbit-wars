# case7 「ルールベースに勝つ」ループ — iter02 RESULT

時刻: 2026-06-03 02:33 (cron tick 3)

## やったこと
1. DVC blocker 解決 → BC weights (case9_per_planet, 12.5MB) 取得・case7 model に
   loaded=133/missing=0 でロード確認。
2. **BC warm-start + KL anchor (kl_beta 0.1)** で学習 (ratio shaping, curriculum
   noop→baseline_jax_lite, iters14/ep6/horizon160)。
3. **参照点を測定**: 生 BC weights (case9 pre-RL) を直接 10戦 vs baseline_v1。

## ★最重要の発見
**生 BC weights (imitation/case9_per_planet, RL 前) ですら vs baseline_v1 = 0/10。**

→ これまでの「RL が伸びない」は誤った問題設定だった。
   **出発点 (BC=模倣学習モデル) が既に baseline_v1 に 0/10**。
   RL は良い方策を壊しているのではなく、**0/10 の方策から始めている**。
   memory `imitation/case1 0/100` とも整合 (模倣系は v1 に勝てていない)。

## Numbers (BC warm-start RL, vs baseline_jax_lite 学習中)
| iter | opp | win | reward | bc_kl |
|---|---|---|---|---|
| 0-1 | noop | 0.50, 0.50 | -0.06, -0.05 | 0.07-0.08 |
| 2 | lite | 0.167 | -0.49 | 0.11 |
| 3-5 | lite | 0.0, 0.0, 0.0 | -0.68〜-0.86 | 0.11-0.13 |
| 6-9 | lite | 0.167, 0.0, 0.333, 0.333 | … | 0.14-0.23 |
| 10-12 | lite | 0.167, 0.167, 0.167 | … | 0.08-0.23 |

- BC ありで vs lite が 0.33 を touch (BC なしの 0-0.25 ceiling よりやや上) だが
  **breakthrough なし**。reward は依然マイナス。
- bc_kl が 0.07→0.23 と増加 = 方策が BC から drift (anchor 0.1 はやや弱い)。

## Diagnosis
- 問題の本体は **RL ではなく出発点の弱さ**。BC(case9) も baseline_jax_lite も
  baseline_v1 も「同じ per-planet 模倣方策の延長線」で、v1 の rule 的精度に
  到達していない。10-15 iter の CPU RL では大差を埋められない。
- baseline_jax_lite は v1 "相当" の近似で、しかも reward 符号で見ると学習相手として
  かなり強い (win 0-0.33)。train(JAX lite)/eval(本物 v1) ギャップも残る。

## ★最終測定 (BC-RL 完走後)
| model | vs baseline_v1 (10戦) |
|---|---|
| 16-iter (BCなし, iter01) | 0/10 |
| **生 BC (case9 pre-RL)** | **0/10** |
| **BC-RL 14 iter (iter02)** | **0/10** (best.pt 正変換確認) |

3 つとも 0/10。**出発点が 0/10 で、小規模 CPU RL では動かない**ことが確定。
(注: best.pt は worktree の `data` symlink 経由だと見えず、main repo 絶対パスで要参照。
今回 1 度誤って stale weights を eval する取り違えあり → 絶対パス運用を徹底。)

## NEXT ACTION (iter03 候補、優先順)
1. **BC-RL 最終 model を 10戦 vs baseline_v1** で測定 (0/10 から動いたか確定)。← 完走後すぐ
2. 出発点が弱いのが主因なら、**学習相手を baseline_jax_lite から「より弱い段階」へ**:
   3 段 curriculum noop → (弱め) → lite で勝てる経験を増やす。または shaping を
   勝ちに直結する terminal 寄りへ (time_bonus 小)。
3. **anchor 強化** (kl_beta 0.1→0.3) で BC からの drift を抑え、勝ち方を保持。
4. 行き詰まり継続なら **web search** で "PPO vs scripted opponent sparse reward
   self-play warmstart" 系の知見を収集 (loop 指示)。
5. 速度: CPU で lite は ~120s/iter と重い。GPU 利用可 (認証不要方針) なので、
   伸びる方向が見えたら RunPod で iter 数を一気に積む選択肢。
