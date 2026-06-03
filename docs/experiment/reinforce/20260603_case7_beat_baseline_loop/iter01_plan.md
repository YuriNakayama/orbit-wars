# case7 「ルールベースに勝つ」ループ — iter01 PLAN

開始: 2026-06-03 01:39 (/loop 30m、job 0d2dff24)

## ゴール
reinforce/case7 (PFSP self-snapshot pool × case5 ratio shaping) を、追加学習 +
学習ロジック改善で **本物 baseline_v1 (rulebase/case1) にローカル対戦で勝てる**
状態にする。検証は最小 (10戦級)、GPU/認証フリー、行き詰まれば web search。

## 制約
- 数十対戦級の大規模検証は避ける (10戦で方向性だけ見る)。
- best.pt(npz) → `jax_to_torch` → `policy/weights.pt` → `eval_vs_baseline` が
  本物評価経路。CHALLENGER は `rl_v7` に修正済み。
- 追加学習は `training.resume_from: <best.pt>` で継続可 (round-trip bit 一致確認済)。

## 出発点 (iter01 measure)
- 16-iter ローカル run (`local_20260602T152240Z`) の best.pt を変換し 10戦:
  **vs baseline_v1 = 0/10 (win_rate 0.0)**。
- memory `project_reinforce_case6_live_eval` と一致 (PFSP は JAX self-play で
  伸びても本物 v1 に 0/10、train(JAX近似)/eval(本物) ギャップ)。

## 仮説 (なぜ 0/10 か / どう改善するか)
- H-a: **学習相手が弱い/近似的**。pool は self_snapshot + baseline_jax_full で、
  本物 v1 を一度も見ていない → 本物 v1 に対する方策が育たない。
  → 対策: opponent に `baseline_jax_lite` (=v1 相当) を curriculum 後半へ入れる、
    または pool に lite を混ぜる。
- H-b: **学習量不足** (16 iter)。resume で iter を積む。
- H-c: **greedy inference の featurizer parity 問題**。eval は torch greedy、
  学習は JAX sampling。まず H-a/H-b で勝率が動くか見てから疑う。

## iter01 でやること (次の tick)
1. baseline 0/10 を記録 (済)。
2. **H-a + H-b**: curriculum を `noop → baseline_jax_lite` に変え (本物 v1 相当を
   主相手に)、16-iter best.pt から `resume_from` で +16 iter 追加学習。
3. 変換 → 10戦 vs baseline_v1。0/10 から動けば H-a 採用、横ばいなら H-c (parity) へ。

## 評価コマンド (最小)
```
uv run python -m pipeline.reinforce.case7.training.jax_to_torch --best <best.pt> \
  --out pipeline/reinforce/case7/policy/weights.pt
uv run python -m pipeline.reinforce.case7.evaluation.eval_vs_baseline \
  --episodes 10 --baseline baseline_v1 --out /tmp/eval.json
```
