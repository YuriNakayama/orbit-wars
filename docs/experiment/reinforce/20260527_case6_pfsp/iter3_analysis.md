# Reinforce/case6 — PFSP f_hard (iter3) ANALYSIS

> 対象: iter3 (H4, f_hard prioritized sampling) / run_id: 20260528-022303__...__510426e__seed0
> 関連: iter3_plan.md / iter3_result.md / iter2_analysis.md / hypotheses.md
> モード: skip mode (replay JSON なし — JAX rollout は in-memory)

metrics.json (100 iter) ベース。H2 (uniform mix) との A/B で f_hard の効果を読む。

## what_worked

1. **f_hard が難敵を優先選択 (設計通り)** — full 選択 52 回 (H2 40 回)。(1−x)^p で
   勝率の低い full に重みが乗り、難敵への露出が約 1.3× に増えた。
2. **vs full の学習が H2 より加速** — last5 0.359→**0.419** (+6pp)、slope +0.0027→**+0.0035**
   (+30%)。難敵集中が「強いルール相手に勝てるよう学習」を押し上げた = PFSP 主手法の有効性。
3. **value_loss 改善** (0.205→0.167) — full への適応で価値推定が締まった。
4. **過去自分は依然上回る** (vs self_snapshot 0.834) — pool snapshot への対応は維持。

## where_to_focus_next (H5 / コスト)

- **vs full 0.42 でまだ伸びしろ**: f_hard で改善したが頭打ちではない。
  → **H5 (f_var=x(1−x))** と A/B。full が強すぎて (1−x)^p の重みが full に張り付き、
    勾配が飽和している場合、同レベル優先 (f_var) の方が安定して伸びる可能性。
- **entropy 52 とやや高い**: full に勝ち切れず探索継続中。H5 で勝率が中間に寄れば収束想定。
- **コスト最重要課題**: f_hard は full 偏重で 92分/$2.1 (H2 の 56分/$0.70 比 1.6×・3×)。
  H5 では (a) iterations 60-80 に削減、(b) priority_p を下げて full 偏重を緩和、
  (c) RTX 4090/3090 が空くまでリトライ ([[project_reinforce_self_snapshot_cost]])。

## why_not_yet_conclusive (n<300)

- vs full last5=0.419 は H2 (0.359) を上回るが、いずれも win_rate は相手構成依存の相対値。
  「Kaggle で通用する絶対強度」は別問題。PFSP 系 (H4/H5) の最良 weights で 300 戦
  (vs baseline_v1 / full / rl_v3) を実施して初めて確定。
- H4 単独では「f_hard > uniform」のトレンド差 (last5 +6pp) は seed 1 本の結果。
  断定には seed 反復 or 300 戦が必要。

## NEXT ACTION

1. H5 (f_var=x(1−x)) を **コスト軽量版** (iter 60-80 / 4090 優先) で実装・実行し f_hard と A/B。
2. f_hard の full 偏重を抑えるため H5 では priority_p を下げる or full の最大選択率に上限。
3. H4 or H5 の最良 iter で 300 戦評価 → PFSP 系の絶対強度を確定。
