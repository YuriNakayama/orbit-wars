# Hypotheses — imitation/case5 ship_prediction

> 作成日: 2026-05-06
> 最終更新: 2026-05-06
> 状態: in_progress
> 最大 iteration: リスト完了後も deepen で継続 (paused / user stop まで)
> 主要メトリクス: 学習中 val_acc / val_loss + 30 ep self-play 挙動確認 (300 ep 評価は実施しない)
> 既定 episode 数: 30 (挙動確認のみ、n<300 で結論は出さない)

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- skill rating は使わない (project rule)

### 分析
- **300 対戦による評価はしない** — 学習中 val_acc / loss curve + 30 ep 挙動確認だけで採否
- n<300 結果で結論を出さない (memory `project_imitation_case1_phase3`) — 30 ep 結果は **inconclusive 固定**、val_acc / loss curve で確定的判断

### 実行
- smoke test (1-episode self-play) を skip ⚠️ — user 確認済み
- dev/test-bot を skip ⚠️ — user 確認済み
- auto-recover loop を使わない — RunPod failure 時は手動介入

### 例外条件
- (なし)

## 仮説リスト (priority 順)

- [ ] (P1) H1: timeline 由来 6 列 (loss_3turn / ttf_norm / min_owned_log / surplus_log / fall_predicted_flag / keep_needed_log) を case1 featurizer に追加し PLANET_FEAT_DIM 11→17 で BC training — rulebase case6 の効力 (vs v4 64%) を imitation に取り込む。plan.md 本線
- [ ] (P2) H5: focal loss α=0.75 + class weight (case1 Phase 2 の breakthrough 構成) を case5 でも適用 — memory `project_imitation_case1_phase2_breakthrough` の知見を timeline feature と組み合わせ
- [ ] (P2, depends on H5) H6: NO_OP minority oversample + focal loss combo — memory `project_imitation_case1_phase3` の延長線。oversample 単独は効果薄なので focal と combo
- [ ] (P3, depends on H1) H2: 6 列の subset (loss_3turn + ttf_norm + min_owned_log のみ 3 列) ablation — H1 でフル投入後、何が効くかの切り分け
- [ ] (P3, depends on H1) H3: auxiliary head: planet 残存 ships 予測 (BC main loss + future ship-residual regression) — BCVA 系。表現学習補助、メイン policy は変えない
- [ ] (P3) H4: auxiliary head: planet ownership flip 予測 (next 5 turn で owner 変化するか categorical) — future state prediction の categorical 版
- [ ] (P3) H7: vs 評価対戦相手を v1 → {v1, v4, v5} 3-way へ拡張 — memory `project_case5_validation` で v5 ≠ v1 なので評価母集団広げる (※本 hypotheses.md は 300 ep skip だが、H7 は逆に評価系の拡張を主張する仮説)
- [ ] (P3, depends on H1) H8: timeline horizon を 30 turn → 10 / 20 turn と短縮した ablation — 計算コスト vs 精度のトレードオフ

## Iteration log

(各 iter 完了時に experiment-execution / experiment-analysis が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|---|---|---|---|---|---|---|---|---|

## 参考 (References)

- [Lux AI Season 3 — Imitation Learning 3rd place writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution) — 類似 RTS タスクでの IL 上位解法 (詳細未読、必要に応じ深掘り)
- [Asking for Help: Failure Prediction in Behavioral Cloning (BCVA)](https://arxiv.org/abs/2302.04334) — 共有 encoder + 補助 head (V(s) / failure prediction) で表現学習。H3 / H4 の理論的根拠
- [Model-based Behavioral Cloning with Future Image Similarity Learning](https://proceedings.mlr.press/v100/wu20b/wu20b.pdf) — auxiliary objective として future state prediction を使う先行研究
- [Game AI Research with Fast Planet Wars Variants (Lucas, 2018)](https://arxiv.org/pdf/1806.08544) — Planet Wars タスクでの forward model / timeline simulation の参考
- [Batch-balanced focal loss (Karpinski 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/) — focal + balanced sampling の hybrid 解。H5/H6 combo の根拠
