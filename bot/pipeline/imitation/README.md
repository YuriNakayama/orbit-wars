# Imitation Learning Cases

過去 Kaggle 上位リプレイ (index: `data/lake/kaggle_episodes/matches/index.parquet/`, payload: `s3://orbit-wars-dvc-286854171013/replays/kaggle/`) からの **行動クローニング (BC)** で
動く PyTorch エージェント群。各 case は独立提出パッケージで、`rulebase/case1` (`baseline_v1`, LB897) 相手に
1v1 勝率 ≥ 50% を目標に改善を重ねる。

## Action 表現

case4 / case8 を除き、以下の 3-head 構造を共通化:

- `from_planet` 多選択 (どの自軍 planet から発射するか、my_planet_mask に限定した multi-hot BCE)
- `target_template` 分類 (各 source ごとに発射先テンプレートを CE)
- `ships_bucket` 分類 (25/50/75/100% の 4 段階を CE)
- `angle` は `aim_with_prediction()` で決定論的に再構成 (学習対象外)

case4 / case8 は「per-source × CAND_K candidate categorical」 (Kaggle tutorial 由来) head に差し替え。

## Status table

| Case | レジストリ名 | featurizer (planet × global) | バックボーン | 役割 / 主な変更点 |
|------|-------------|---:|---|------|
| case0 | `il_v0` | 8 × — (synthetic) | `Linear(8,16)→ReLU→Linear(16,4)` | RunPod E2E smoke 専用 (commit→push→S3→DVC pull の周回検証)。モデル改善対象外 |
| case1 | `il_v1` | 11 × 6 | DeepSets | **canonical baseline**。Phase 2 で iter9 が 5/100、再評価 0/300。target diversity が残課題 |
| case2 | `il_v2`, `il_v2_phase1` | 18 × 11 / 33 × 16 | DeepSets / Graph U-Net | case1 から planet 7 + global 5 列追加 (sun_dist, support_density, threat_density, comet, phase 等)。`il_v2_phase1` は Graph U-Net 移行版 |
| case3 | `il_v3` | 35 × 20 | Graph U-Net (kNN k=8, hidden=128, TopK pool 3 段) | case2 + 観測履歴ベースの時系列 3+4 列。HistoryState を per-match 保持 |
| case4 | `il_v4` | 35 × 20 (case3 と同一) | Graph U-Net (case3 と同一) | バックボーンは触らず、出力 head を **per-source × CAND_K candidate categorical** に差し替え (Kaggle tutorial 派生)。slot 0 = no-op |
| case5 | `il_v5` | 17 × 6 | Graph U-Net | case1 の 11 dim + **rulebase/case6 から portage した ship-prediction timeline 6 列** (loss_3turn, ttf, min_owned, surplus, fall_predicted, keep_needed) |
| case6 | `il_v6` | 17 × 6 (case5 同一) | **Graph Attention U-Net** (multi-head 4, edge feat) | case5 の GraphConv (uniform mean) を attention に置換。pairwise (dx, dy, dist, ship_log_diff, tgt_is_enemy, tgt_is_neutral) を attention bias に注入 |
| case7 | `il_v7` | 17 × 6 (case5/6 同一) | **Set Transformer** (ISAB m=16 + PMA) | kNN グラフを廃止。ISAB で O(P²)→O(P·m)、PMA で template ごとに learnable query を持つ cross-attention target head |
| case8 | `il_v8` | 35 × 20 (case3/4 同一) | Graph U-Net (case3 と同一) | case4 の candidate head 系を引き継ぎ、ship-prediction を取り込んだ feature engineering branch。iter13 時点で in-memory dataset 化 + EMA/dropout/grad_clip/lazy/focal を整理 |

## 系譜

```
case0  独立 (RunPod 基盤検証)

case1 (canonical baseline)
  └─ case2 (planet/global 特徴拡張) ── il_v2_phase1 で Graph U-Net 移行
       └─ case3 (時系列 + Graph U-Net)
            └─ case4 (head 差し替え: candidate head)
                 └─ case8 (case4 + ship-prediction / dataset・最適化整理)

case5 (case1 + rulebase/case6 timeline 6 列)
  ├─ case6 (case5 backbone を Graph Attention 化)
  └─ case7 (case6 から kNN を捨て Set Transformer 化)
```

## Conventions

- `case<N>/policy/` が submission code (Kaggle に同梱)
- `case<N>/training/` は `.submitignore` で除外、ローカル開発のみ
- `case<N>/evaluation/` は `src/evaluation/{metrics,vs_baseline}.py` 経由
- `policy/{geometry,decoder}.py` は case 間で 100% 重複しているが case 独立原則のため許容
- cross-case import は禁止 (case 内で完結)

## モデルバージョン命名規則 (case 内バージョニング)

| ファイル | 役割 |
|---------|------|
| `policy/weights.pt` | **canonical**。Kaggle 提出に同梱される最新採用版。`dev/{vast,runpod} promote` で上書き |
| `policy/weights_iter<N>.pt` | iteration 履歴。`<N>` は学習回数の連番 (0 = baseline、1..N = 改善 iter) |
| `policy/weights_iter<N><suffix>.pt` | 同 iter 内 variant (例: `iter10b` = iter10 の `aggregate="mean"` 版) |
| `policy/weights_smoke.pt` | smoke test 用 (1 epoch)。case6/case7 で使用 |

`agent.py` は常に `weights.pt` を読み込む。過去 iter を再評価する際は明示的に
`--weights pipeline/imitation/case<N>/policy/weights_iter<N>.pt` を指定する。

詳細: `docs/plans/refactor-directory/`
