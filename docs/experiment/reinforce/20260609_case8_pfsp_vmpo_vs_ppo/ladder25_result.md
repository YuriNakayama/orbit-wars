# case8 PFSP V-MPO — ladder25 (no_op_bias 1.0→0.0) RESULT

> 関連: vmpo_ladder25.yaml / strict_win_structural_cause.md / strict_win_cases_diff.md
> run_id: 20260617-034316__feature-poc-v-mpo__e302058__seed0 / commit: e302058 / case: case8
> pod: qlhg7xrfdatim8 (RTX4090 SECURE) / runtime: 279min (4.65h) / exit 0 (self-destruct確認)
> resume: ladder22 best.pt (full 0.86)

## Summary
no_op_bias 1.0→0.0 の clean な1変数 A/B (他は ladder22 と byte-identical)。仮説「地力(full)↑
が strict勝率↑の最短」を地力面で支持: full は 0.781→**0.875** (iter10, キャンペーン最高タイ)
に到達し、NO-OP罰を完全撤廃しても過小発射崩壊は起きなかった。strict held-out は iter20-50 で
0.031/0.016/0.016/**0.047** と nonzero を持続 (iter50 で full 0.859 と strict 0.047 を同時に
キャンペーン最高帯で達成) も、iter60/69 で 0 に振動回帰。**「稀勝利→安定勝利」の壁 (壁③) は
越えられず**。no_op_bias=0 は地力中立〜微正・strict中立の clean なレバーと確定。

## Numbers (held-out, 固定seed 777000, 64戦)

| iter | full | strict | 備考 |
|---|---|---|---|
| 0 | 0.781 | 0.000 | ladder22 base (no_op_bias=0 適用直後) |
| 10 | **0.875** | 0.000 | full キャンペーン最高タイ |
| 20 | 0.797 | 0.031 | strict 初勝利 (2/64) |
| 30 | 0.781 | 0.016 | strict 持続 (1/64) |
| 40 | 0.828 | 0.016 | strict 持続 (1/64) |
| 50 | 0.859 | **0.047** | full+strict 同時キャンペーン最高帯 (3/64) |
| 60 | 0.859 | 0.000 | strict 振動回帰 |
| 69 | 0.797 | 0.000 | 終端 |

- strict held-out 平均 (eval点8つ): ~0.020。max 0.047 = ladder21 と並ぶキャンペーン最高。
- full 平均: ~0.82、max 0.875。**ladder22 (no_op_bias=1, full ~0.86) と同等** → no_op_bias=0
  は地力を落とさない。
- 学習健全性: self_snapshot win 0.53-0.68 (PFSP ~0.5-0.6 維持)、entropy 38-42 (崩壊なし)。
- A2 skip: 偶数iterの T0=0 素strict段 (win 0.01-0.02) を全て更新スキップ → 方策poison回避を確認。

## Diagnosis
- **壁① (0→稀勝利) は fine strict ladder で既に越えていた** (strict_win_structural_cause.md):
  ladder25 は ladder21→22 由来の fine-ladder (T0刻み15, T0=110段) + degenerate-batch guard
  (skip A + adv_std_floor B) を継承。これが strict held-out を 0.016-0.047 帯に乗せた。
- **no_op_bias=0 の寄与 = 地力の保存**: full を 0.875 まで押し上げ、ladder22 比で地力を落とさず
  に壁①の効果を維持。NO-OP罰撤廃の過小発射リスクは顕在化しなかった (entropy健全)。
- **壁③ (稀勝利→安定) は未踏のまま**: iter50 で full 0.859 + strict 0.047 を同時達成しても
  iter60 で strict=0 に振動。strict勝利は full 絶対値でなく方策状態の確率的揺らぎ依存で、
  fine-ladder+guard+地力max でも 3/64 が天井。素strict の開幕戦略エッジ (持ち船3倍handicapでも
  ~4%, ladder20) は構造的で、緩和段の転移だけでは常勝に届かない。

## Decision
- 採否: **inconclusive (strict常勝目標に対して)** / no_op_bias=0 自体は **adopted (clean中立レバー)**。
- 結論の核: **ladder/guard/reward/no_op_bias 等の RL ハイパーレバーは出尽くした**。fine-ladder+guard
  が「0→稀勝利」を確立し、その後の全レバー (terminal_scale, dense boost, handicap, reverse curriculum,
  no_op_bias) は strict 稀勝利帯 (0-0.047) を動かせなかった。**壁③ には質的に別レバーが必要**。
- 次の一手: **strict-BC bootstrap** (in-JAX strict の開幕 N手を模倣 → KL-anchor付き RL)。素strict の
  零分散 (勝てない→勾配なし) を模倣の教師信号で迂回する唯一の未踏レバー。大規模ビルドのため
  ユーザー判断を要する (過去の human-replay BC は失敗したが、これは in-JAX strict 模倣で別物)。

## Artifacts
- model: data/output/models/reinforce/case8_vmpo_ladder25/runs/20260617-034316__.../best.pt (S3/DVC)
- metrics: 同 runs/.../metrics.json (history 70 iter)
- pod: qlhg7xrfdatim8 自己破棄確認 (`dev/runpod ps` = No active pods)
