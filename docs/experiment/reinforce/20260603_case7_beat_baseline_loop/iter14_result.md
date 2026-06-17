# case7 ループ — iter14 RESULT: f_hard 汚染を発見・修正して pure self-play 再実行

時刻: 2026-06-03 15:02 (cron tick 21)

## ★config の落とし穴を発見
iter14 (f_hard ratchet) 起動後、iter2-4 に **opp=baseline_jax_full (reward -2.0 飽和)**
が混入していた。原因:
- `priority: f_hard` は `selector.set_entries(pool.models(), include_full=True)`
  (train_jax.py:685) で **late_full_prob=0.0 を無視して常に baseline_jax_full を
  候補に含める**。→ 純 self-play にならず、有害な full が混ざる (iter13 と同じ劣化リスク)。
- 打ち切り、修正版を再実行。

## 修正: pure self-play
- `priority: uniform` + `late_full_prob: 0.0` → in-loop 選択が**常に pooled self
  snapshot** を選ぶ = 真の pure self-play。full 混入なし。
- iter14b: iter12 BEST から resume、uniform pool (snapshot_every2, cap6)、h500、18 iter。
- 起動確認: iter0 noop のみ、full 不在。

## 教訓 (config)
- **pure self-play がしたいなら `priority: uniform` + `late_full_prob: 0`**。
  `f_hard` は include_full=True 固定なので強相手 (full) が必ず入る → 非飽和を保てない。
- これは memory `project_reinforce_unbeatable_opponent_harmful` の具体的な踏み方。

## NEXT (iter14b)
- pure self-play 18 iter 完走 → 10戦 vs rl_v0 (0.90 維持/超えるか)。
- self-play ratchet で「過去の自分」を倒し続け漸進強化できるかの検証。

## ★iter14b 最終測定 (15:50) — self-play 続行も rl_v0 で劣化
| 相手 | iter14b (pure self-play +18) | iter12 (self-play 16) |
|---|---|---|
| rl_v0 | **4/10 (0.40)** | **9/10 (0.90)** ← ベスト |
| baseline_v1 | 0/10 | 0/10 |

- reward は健全 (非飽和 -0.6〜+1.33、full 混入なし) だったのに **rl_v0 0.90→0.40 に劣化**。
- 原因: **self-play は「現在の pool (直近の強い自分)」に最適化する。pool が ratchet で
  強くなるほど自分の style に特化 → rl_v0 のような外部相手への汎化を失う** (overfitting
  to self-play distribution)。
- = **self-play を回しすぎると外部相手で弱くなる**。iter12 (16 iter) がたまたま汎用的に
  強い点に着地、iter14b (+18) は self 特化へ overshoot。

## ★ループの核心的知見 (3 つの劣化要因)
rl_v0 0.90 (iter12) を起点に:
1. **lite を学習相手に追加** (iter13) → 0.40 (reward 飽和の害)。
2. **self-play を続行** (iter14b) → 0.40 (self 特化 overfit の害)。
→ どちらも劣化。**iter12 の "16 iter self-play" が sweet spot**。狭い。

## 確定: ベスト model = iter12
- `local_20260603T031704Z/best.pt` (self-play ladder 16 iter、rl_v0 0.90)。
- これ以上いじると劣化する。小規模 RL の到達点。
- v1 越えは scale (より大きく多様な pool + 長時間) が要る = Generals.io 路線。
  小規模では iter12 が天井。

## NEXT
- iter12 を確定ベストとして保全 (policy/weights.pt に変換済を確認)。
- これ以上の小規模追加学習は劣化リスク高 → 学習レシピ探索は飽和。
- 残るは scale (GPU) のみ。ユーザー判断待ち。
