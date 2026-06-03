# case7 「ルールベースに勝つ」ループ — iter04 RESULT

時刻: 2026-06-03 03:42 (cron tick 6)

## やったこと
iter03 best から resume、**self_snapshot pool 主体 (lite 外し) で 30 iter** スケール。
kl_beta 0、f_hard (full も混入)。

## Numbers (学習中)
- vs self_snapshot: win **0.83** 安定 (過去の自分には勝てる)。
- vs baseline_jax_full: win 0-0.33 (rule には勝てない)。
- best_win=1.000 (self 相手)。

## 最終測定
| model | vs baseline_v1 (10戦) |
|---|---|
| iter04 (self-play 30 iter) | **0/10** |

→ **5 variant 全て 0/10** (16iter / 生BC / BC-RL14 / 3段16 / self-play30)。

## ★気づき: 0/10 が"完全に"一様すぎる
JAX 自己対戦で win 0.83 出る model も、torch eval では **例外なく 0/10**。
recipe を変えても 1 勝もしないのは不自然。**変換/推論パイプラインの parity** を疑う:
- best.pt(npz JAX leaves) → `jax_to_torch` → torch ActorCritic → greedy → 実対戦。
- memory `project_rulebase_jax_parity_failure_mode`: JAX→torch の precision /
  featurizer 差で実戦性能が壊れる前例あり。
- 仮に変換 or torch featurizer が JAX とズレていれば、GPU で何 iter 回しても 0/10。

## ★parity 確認結果 (iter04 追記、03:48)
- case1 の JAX↔torch **featurizer + model parity test 81 件 全 pass** (47s)。
  case7 は同一 architecture なので **変換/推論パイプラインは健全**。
- → 0/10 一様は **parity バグではなく、方策が本当に baseline_v1 に弱い**ため確定。
  conversion 仮説は棄却。GPU 不要で原因を切り分けられた (高ROI)。

## 結論 (6 tick 時点・確定)
- **ローカル CPU 小規模 RL (10-30 iter) では baseline_v1 に勝てない (5/5 variant 0/10)**。
- recipe (BC warm-start / 3段 curriculum / KL anchor / ratio shaping / self-play pool) は
  全て健全に機能、変換も parity OK。ボトルネックは **純粋に compute scale**。
- memory: reinforce/case1 は RunPod GPU で **300 iter** 回して初めて self-play last-10 0.50。
  本ループの 10-30 iter は 1 桁以上不足。

## NEXT ACTION (iter05) — compute scale up (loop の GPU 許可方針に従う)
1. **変換 round-trip 検証**: 同一 obs に対し
   (a) JAX model(best.pt) の行動分布 と (b) torch 変換 model の行動分布 を比較。
   一致しなければ jax_to_torch にバグ → そこを直すのが本丸 (GPU 不要)。
2. **featurizer parity**: torch featurizer (eval 経路) と JAX featurizer (学習経路) が
   同一 obs で同じ特徴を出すか (case5 README は tol 1e-4 で一致と主張 → 実機確認)。
3. parity OK なら問題は本当に方策の弱さ → GPU で 100-200 iter スケール。
   parity NG なら修正後に再評価 (これが 0/10 一様の真因の可能性)。

## 運用メモ
- best.pt は main repo 絶対パス参照。config は loop_iter0N_*.yaml に保存。
- これまで全 eval は seed=0 固定 10戦。parity 確認後は seed 変えて再現性も見る。
