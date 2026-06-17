# iter1 result: H1 handicap curriculum — REJECTED

run_id: 20260608-054911...90444c5 / GPU: RTX 4090 / ~10min / config: h1_handicap.yaml

## 設定
3-stage: noop(0-2) → baseline_jax_lite(3-13) → baseline_jax_full(14-19)。
h500/batch32, ratio shaping coef=1.0, from scratch (BC warm-start なし)。

## 結果
| stage | iters | opp | win 推移 | 平均 |
|---|---|---|---|---|
| noop | 0-2 | noop | 0.72→0.78 | 0.74 |
| handicap | 3-13 | baseline_jax_lite | 0.16/0.28/0.28/0.19/0.09/0.31/0.16/0.28/0.22/0.31/0.22 | **~0.22** |
| target | 14-19 | baseline_jax_full | 0.25/0.38/0.28/0.34/0.31/0.38 | **~0.32** |

- entropy: noop期 46-50 → lite突入で **12→9 に急落**（policy collapse 気味）。
- **lite stage で学習 trend なし**（0.1-0.3 で振動、上昇せず）。handicap が機能していない。
- full stage は ~0.32 で baseline run の ~0.22 よりやや高いが **n=32 のノイズ域**、学習による改善ではない。

## 結論: REJECTED
- **opponent 難度の調整(handicap)は効かない**。baseline_jax_lite ですら untrained agent には実質勝てない相手で、勾配 foothold を作れず entropy collapse。
- 教訓: ボトルネックは **相手の強さでなく学習信号の質**。sparse terminal + ratio shaping だけでは rulebase を倒す tactical behavior を学べない。
- → 次は **H2 reward densification**（Minimax reward: case8 scoring を dense penalty 化）に注力。opponent をいじるより reward を変える。

## メトリクス
- vs baseline_jax_full (学習中 in-JAX, n=32): ~0.32（前 baseline ~0.22 と有意差なし）
- 最終 ckpt の外部 paired 評価は省略（学習中 win が改善していないため採否は明らか）
- コスト: ~$0.14（4090 12分）
