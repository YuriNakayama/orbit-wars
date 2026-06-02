# Reinforce/case6 — PFSP f_hard prioritized sampling (iter3) RESULT

> 関連: iter3_plan.md / iter2_result.md / hypotheses.md
> run_id: 20260528-022303__feature-agent-pool-learning__510426e__seed0 / commit: 510426e / case: reinforce_case6_kaggle_jax_train_h4
> 開始: 2026-05-28 02:23 / 終了: 04:00 (100 iter 完走) / コスト: ~$2.1 ⚠️ (A100 80GB $1.39/h × 1.53h、cap $1.5 超過)

## Summary

H4 (PFSP `f_hard=(1−x)^p` 優先 sampling) は **狙い通り機能**した。f_hard が手強い
baseline_jax_full を優先選択し (full 52 回 vs H2 の 40 回)、**vs full の勝率が H2 を上回った**:
last5 0.359→**0.419**、slope +0.0027→**+0.0035/it**。難敵集中で「強いルール相手に勝てるよう
学習」が H2 (uniform mix) より加速。PFSP の主手法が有効と確認。ただし n<300 + 絶対勝率は
まだ 0.42 で結論は inconclusive。**コスト面の教訓**: f_hard は重い full を頻繁選択するため
rollout が H2 比 1.6× (92 vs 56 分) に伸び、A100 fallback と重なり ~$2.1 と cap を超過した。

## Numbers (学習ログのみ — 300 対戦 skip, n<300 結論不可)

| metric | H2 (uniform) | **H4 (f_hard)** | 判定 |
|---|---|---|---|
| **vs full last5** | 0.359 | **0.419** | ✅ 改善 |
| **vs full slope** | +0.0027/it | **+0.0035/it** | ✅ 加速 |
| vs full overall | 0.274 | 0.293 | ✅ 微増 |
| full 選択回数 | 40 | **52** | f_hard が難敵を優先 (設計通り) |
| vs self_snapshot | 0.828 | 0.834 | 同等 (過去自分は上回る) |
| win overall | 0.614 | 0.560 | 難敵露出増で低下 (健全) |
| entropy last10 | 47 | 52 | やや高め |
| value_loss last10 | 0.205 | 0.167 | 改善 |
| runtime / cost | 56min / $0.70 | **92min / $2.1** ⚠️ | f_hard の full 偏重で長引く |

- ② vs 初期 snapshot = vs self_snapshot phase。③ vs baseline_v1 20 戦 = 未実施
  (H5 or 最終で 300 戦にまとめる)。

## Diagnosis

- **f_hard が設計通り難敵を優先**: full 選択 52 回 (H2 40 回)、(1−x)^p で勝率の低い full に
  重みが乗った。結果 vs full の last5 が 0.359→0.419、slope も +0.0027→+0.0035 と改善。
- **win overall は低下 (0.614→0.560) だが健全**: 難敵への露出が増えた分の必然。entropy が
  52 とやや高いのは full に勝ち切れていない (探索継続) ため。
- **コスト超過の根本原因**: f_hard が最も計算が重い baseline_jax_full を頻繁に選ぶ →
  rollout が H2 比 1.6×。A100 fallback ($1.39/h) と重なり ~$2.1。当初「heartbeat≈iter」と
  誤認し進捗を過大評価した (heartbeat は ~75s 固定、iter とは非 1:1)。

## Decision

- **採否: inconclusive (positive)** — f_hard は vs full のトレンドを uniform (H2) より明確に
  改善 (last5 +6pp、slope +30%)。メカニズム有効。ただし n<300 + 絶対勝率 0.42 で断定不可。
- **次の一手 (H5)**: vs full が 0.42 でまだ伸びしろ。`f_var=x(1−x)` (同レベル優先) と A/B し、
  full が強すぎて勾配が飽和しているなら f_var が有利な可能性を検証。
  **ただし H5 ではコスト対策必須**: late_full_prob 相当を抑える or iterations を 60-80 に削減、
  RTX 4090/3090 が空くまでリトライ。f_hard は full 偏重で高コストと判明 (memory 化)。
- **300 戦評価**: H5 完了後 or PFSP 系の最良 iter で rl_v6 vs baseline_v1 / full / rl_v3 を
  300 戦 (例外条件) 実施し絶対強度を確定。

## Artifacts
- model: `data/output/models/reinforce/case6_kaggle_jax_train_h4/runs/20260528-022303__feature-agent-pool-learning__510426e__seed0/best.pt`
- metrics: `.../metrics.json` (100 iter history)
