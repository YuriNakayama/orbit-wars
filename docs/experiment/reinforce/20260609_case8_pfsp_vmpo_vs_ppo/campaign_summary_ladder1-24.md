# strict_v1 攻略 campaign 総括 (ladder1-24, 2026-06-17)

## 成果と未解決
| 指標 | 結果 |
|---|---|
| **held-out full** (movement detector) | **0.875 ピーク** (ladder24 iter0) / 0.86 安定 (ladder22) — 強い汎用エージェント確立 |
| **held-out strict_v1** (本物・序盤含む) | **~0% (0-3/64)** — 24 run 全て noise床、未突破 |

## strict 攻略で試して棄却した全レバー
| レバー | run | 結果 |
|---|---|---|
| curriculum形状 (T0/ε 時間窓) | ladder1-13 | 素strict 0% |
| 逆カリキュラム (warmup後退) | ladder14-16 | 0% + 実行コスト律速 |
| 対戦量↑ (mix_strict 0.85) | ladder17 | 0% (zero-variance) |
| aim修正 (intercept) | ladder18 | **full 0.828 達成**、strict 0% |
| reward整形 (terminal_scale) | ladder13,19 | 0% |
| degenerate-batch guard (A skip) | ladder19,21,22 | poison阻止✓ も strict 0% (skipは学習でない) |
| no_op_bias↓ (C, over-fire是正) | ladder21,22 | **地力↑ (full→0.86)**、strict 0% |
| handicap (material 3倍) | ladder20 | ~4% (strict優位は material でなく戦略) |
| dense信号 on T0=0 (B) | ladder23,24 | 0% (territory比は学ぶが勝利に変換せず) |

## 確定した診断 (root cause)
1. **断崖は素strict序盤 (turn 0-110)** に局在。中盤以降 (T0≥140) なら 0.5-0.8 勝てる。
2. **strict の序盤優位は「戦略/位置」**であり material でも aim でもない (handicap 3倍でも~4%)。
3. **zero-variance**: 素strict に ~0% 勝利 → 増幅すべき勝利エピソードが無く方策勾配が
   立たない。reward整形/dense/skip いずれも「勝てない局面に勝利信号を作る」ことは不可。
4. degenerate poison (entropy崩壊) は A2 skip で解決済だが、skip では学習しない。

## 残る未投入の本筋 (将来)
campaign で唯一未実施の文献本命 = **strict-BC bootstrap (imitation)**: strict_v1 (決定的・
安価・in-JAX教師) の state→action を BC で直接模倣し warm-start → KL anchor付き RL洗練
(AlphaStar/Lux/MimicBot の実証パターン)。過去のBC失敗は「人間棋譜」で strict-BC は別。
effort 大 (BC データ生成 + 学習) だが、「勝ちを探索で発見」でなく「教示」する唯一の道。

## 推奨
- **full 0.875 / 0.86 を成果として確定** (ladder24/22 best.pt)。movement detector 系には強い。
- strict 攻略は「reward/curriculum の調整」では限界。次に投資するなら strict-BC bootstrap
  (大規模) か、ここで一旦区切る判断。
