# case7 「ルールベースに勝つ」ループ — iter03 RESULT

時刻: 2026-06-03 03:00 (cron tick 4 / 1時間経過)

## やったこと
3 段 curriculum `noop → self_snapshot → baseline_jax_lite` + anchor 強化
(kl_beta 0.3) + BC warm-start + ratio shaping、16 iter。

## Numbers (学習中 win, opp 別)
| iter | opp | win | reward |
|---|---|---|---|
| 0-1 | noop | 0.50 | ~-0.07 |
| 5-8 | **self_snapshot** | 0.167, **0.833**, 0.667, 0.333 | 一部 **+0.26** |
| 9-15 | baseline_jax_lite | 0.167, 0.0, 0.167, 0.167, 0.167, 0.333, 0.0 | ~-0.6 |

- **中間 (self_snapshot) で win 0.83 / reward +0.26** = curriculum は機能、過去の自分には勝てる。
- **lite に移ると 0-0.33 / reward マイナス** = lite (v1相当) は依然壁。

## ★最終測定
| model | vs baseline_v1 (10戦) |
|---|---|
| iter03 (3段, kl0.3, 16 iter) | **0/10** |

これで **4 variant 全て 0/10** (16iter / 生BC / BC-RL14 / 3段16)。

## 結論 (1時間時点)
- 「自分には勝てるが baseline_v1 には勝てない」= **方策の絶対的強さが v1 に届かない**。
- recipe (BC / curriculum / anchor / shaping) は健全に機能しているが、
  **小規模 CPU RL (10-16 iter) では 0/10 の出発点から v1 を越えられない**ことが確定。
- 真のボトルネックは **compute scale** (memory: case1 は 300 iter / RunPod GPU で
  初めて last-10 0.50 に到達)。CPU で lite 1 iter ~100-180s は遅すぎる。

## NEXT ACTION (iter04〜)
1. **scale up**: 認証不要方針なので RunPod GPU で iters を一桁増やす (100-200 iter)。
   ただし memory: self_snapshot/PFSP は rollout 2倍重 + 3090/4090 stockout 注意。
   → まず opponent を軽い構成 (noop→self_snapshot 中心、lite/full は後半少量) にして
     1 run のコストを抑える。est cost を確認してから起動。
2. あるいは **CPU で iter 数を稼ぐ**: lite を完全に外し self_snapshot 主体で
   30-40 iter 回し「自己対戦で強くなった model」が v1 に少しでも勝てるか確認
   (lite が重いボトルネックなので外すと iter 数 2-3 倍稼げる)。
3. eval は引き続き 10戦 (方向確認)。0/10 から 1 でも勝てたら 30-50 戦で再確認。

## 運用メモ
- best.pt は main repo 絶対パスで参照 (worktree data symlink 不安定、取り違え 1 回あり)。
- 全 config は `pipeline/reinforce/case7/configs/loop_iter0N_*.yaml` に保存済。
