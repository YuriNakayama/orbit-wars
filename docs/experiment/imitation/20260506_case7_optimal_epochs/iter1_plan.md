# imitation/case7 — Optimal Epoch Count Diagnosis

> 作成日: 2026-05-06
> 関連:
>   - `docs/experiment/imitation/20260505_case7_model_structure_roadmap/iter1_result.md` (case7 epoch=15, best_epoch=14)
>   - `docs/experiment/imitation/20260505_case7_model_structure_roadmap/iter3_result.md` (case10/11 規模拡大失敗、 under-trained と診断)
> スコープ: case7 (Set Transformer, 現状最良) を **epoch=30** で学習し、 真の best_epoch を観測して最適 iter 数を推定

## 仮説 (Hypothesis)

**case7 (epoch=15) は under-trained**。 best_epoch=14 (= 最終 epoch) で val_loss がまだ下がり続けていた = capacity に対する訓練量不足。 case7 を **epoch=30** で学習すれば真の best_epoch が 20-25 epoch 付近に出現し、 best_val_loss は更に低下 (3.6211 → 3.55-3.60 想定) する。

## 背景 (Background)

iter3 で case10 (Set Transformer 256 XL) / case11 (Pointer Net 256 XL) を試し、 規模拡大が逆効果だったことから:
- train_loss は最終 epoch まで下がり続ける = under-trained 確定
- 真の bottleneck は **規模ではなく訓練量**
- まずは epoch を増やして best_epoch がどこに来るかを観測する診断実験

## スコープ (Scope)

- 変更ファイル: `bot/pipeline/imitation/case7/configs/il_case7.yaml` の `epochs: 15 → 30` のみ
- 副次変更: model.py に factory pattern 導入 (case8/10/11 削除に伴う 1-case 集約)
- データセット / 特徴量変更: なし

## 実装ステップ (Implementation outline)

1. ✅ case8/10/11 削除 (`bot/pipeline/imitation/case{8,10,11}/`、 `bot/tests/pipeline/imitation/case{8,10,11}/` 削除済)
2. ✅ `bot/src/dataset/selfplay/agents.py` から `il_v8` / `il_v10` / `il_v11` 削除
3. ✅ `bot/src/runpod_io/cli.py` の `CASE_DEFAULTS` から case8/10/11 削除
4. ✅ `dvc.yaml` から case8/10/11 stage 削除
5. ✅ `bot/pyproject.toml` から case8/10/11 ignore 削除
6. ✅ `bot/pipeline/imitation/case7/policy/model.py` に **factory pattern** 導入 (`build_model(cfg)` で `cfg.model_type` 分岐、 `set_transformer` / `pointer` をサポート)
7. ✅ `bot/pipeline/imitation/case7/training/train.py` を `build_model` 経由に修正
8. ✅ `bot/pipeline/imitation/case7/configs/il_case7.yaml` に `model_type: "set_transformer"`、 `epochs: 30` 設定
9. ✅ tests に factory 4 件追加 (19/19 pass)
10. ✅ ローカル smoke で factory 動作確認

## 検証方法 (Validation method)

- ローカル: `dev/test-bot` (lint / pytest / mypy 全 green)
- リモート: `dev/runpod train <sha> --case case7 --gpu-name 'NVIDIA RTX A6000' --gpu-name 'NVIDIA GeForce RTX 4090' --image runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204`
- 想定所要時間: preprocess ~30 分 + train (epoch=30, RTX 4090) ~14 分 + 環境構築 ~10 分 = **~55 分**
- 想定コスト: ~$0.65 (case7 iter1 と同等 + 学習時間 +50%)
- 評価: val_loss / val_target_acc / val_from_acc / val_ships_acc を **全 30 epoch** プロット、 best_epoch を観測

## 成功条件 (Decision threshold)

| 結果 | 解釈 | 次 action |
|---|---|---|
| best_epoch ≤ 14 | 既に case7 (epoch=15) で十分、 epoch 数は bottleneck ではない | epoch axis を放棄、 別 axis に |
| **best_epoch = 15-25 + val_loss < 3.62** | **iter 数増で改善** が証明、 規模拡大は epoch 増と組み合わせで再検証可能 | epoch=30 を canonical に、 case10/11 (規模拡大) を再実験 |
| best_epoch = 28-30 + val_loss < 3.62 | epoch 50 まで伸びる可能性、 epoch=50 で再実験 | epoch=50 で再実験 |
| best_val_loss ≥ 3.62 (case7 iter1 比悪化) | seed variance または over-train、 epoch 増は逆効果 | epoch=15 を維持、 別 axis に |

## リスク / 既知の不確実性

- **regression test**: model_type='pointer' は旧 case8 の挙動を再現すべき。 ローカル smoke では shape 確認のみ、 完全な ablation は別実験
- **訓練時間**: epoch=30 は train ~14 分 (epoch=15 の 7 分 × 2)、 完走時間は preprocess 30 分支配
- **対戦勝率**: 本実験の主目的は val_loss 観測。 vs baseline_v1 評価は補助

## 参考 (References)

- iter1_result.md: epoch=15 で best_epoch=14, val_loss=3.6211
- iter3_result.md: case10/11 規模拡大が under-trained で失敗 → epoch 増の必要性
