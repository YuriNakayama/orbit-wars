# imitation/case7 — Optimal Epoch Count Diagnosis (iter2: epoch=50)

> 作成日: 2026-05-06
> 関連:
>   - [`iter1_plan.md`](./iter1_plan.md) (epoch=15→30 仮説)
>   - [`iter1_result.md`](./iter1_result.md) (epoch=30 完走、 best_epoch=28、 plateau 未確定)
> スコープ: case7 (Set Transformer) を **epoch=50** で学習し、 iter1 で未確定だった真の plateau を確定させる

## 仮説 (Hypothesis)

**iter1 (epoch=30) の best_epoch=28 はまだ plateau 直前**。 val_loss は epoch 28 まで明確に低下し続け (3.6211 → 3.5631)、 epoch 29 で 0.014 上振れた。 これは noise の可能性もあるが plateau 開始 (= over-train への移行点) の可能性もある。 **epoch=50 で学習を延長**すれば、 真の plateau (val_loss 下げ止まり epoch + 5 epoch margin で安定) を確定できる。

予想シナリオ:

| シナリオ | 観察される best_epoch | val_loss | 解釈 |
|---|---|---|---|
| A: plateau は epoch 28 付近 | best_epoch ≈ 28-32 | 3.55-3.57 | iter1 で既に最適、 epoch=30 を canonical に |
| B: plateau は epoch 35-40 | best_epoch ≈ 35-40 | 3.50-3.55 | epoch=40 を canonical に |
| **C: plateau は epoch 45+** | best_epoch ≈ 45-49 | 3.48-3.53 | epoch=50 でも未確定、 epoch=80 へ |

## 既存コードの現状 (from iter1)

- `bot/pipeline/imitation/case7/` は 1-case 集約済み (factory pattern、 model_type="set_transformer" / "pointer")
- iter1 の epoch=30 は `il_case7.yaml` の `epochs: 30` で実行
- iter2 では `epochs: 50` に変更、 他は不変

## スコープ (Scope)

- 変更ファイル: `bot/pipeline/imitation/case7/configs/il_case7.yaml` の `epochs: 30 → 50` のみ
- 副次変更: なし
- データセット / 特徴量変更: なし

## 実装ステップ (Implementation outline)

1. `il_case7.yaml` の `epochs: 30` を `epochs: 50` に変更
2. lint / test 確認 (15-19 tests pass)
3. commit + push
4. RunPod launch (case7、 cu1241 image、 A6000/4090 SECURE)
5. 完走後 Phase 7:
   - artifact 取得 + weights.pt 配置
   - epoch 0-49 学習曲線プロット
   - iter1 (15ep) / iter1 (30ep) / iter2 (50ep) 3-way overlay 比較プロット
   - eval vs baseline_v1 50戦 sanity
   - iter2_result.md 作成

## 検証方法 (Validation method)

- ローカル: smoke 不要 (epoch 数のみ変更、 model 構造同じ)、 lint/test は実施
- リモート: `dev/runpod train <sha> --case case7 --gpu-name 'NVIDIA RTX A6000' --gpu-name 'NVIDIA GeForce RTX 4090' --image runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204`
- 想定所要時間: preprocess ~33 分 + train (epoch=50) ~30 分 + 環境構築 ~10 分 = **~75 分**
- 想定コスト: ~$0.85 (epoch=30 比 +$0.20)
- 評価: val_loss / val_target_acc を **全 50 epoch** プロット、 plateau 開始 epoch を観測

## 成功条件 / 採否しきい値

| 結果 | 解釈 | 次 action |
|---|---|---|
| best_epoch ≤ 30 | iter1 epoch=30 で十分、 epoch axis fix | epoch=30 を canonical 維持 |
| **best_epoch 31-45 + val_loss < 3.55** | **iter1 推測 (epoch=30 plateau 未確定) を支持**、 epoch=40-50 が最適 | best_epoch + 5 epoch を canonical に |
| best_epoch 46-49 + val_loss < 3.50 | epoch=50 でも plateau 未確定 → epoch=80 で再実験 | iter3 で epoch=80 |
| val_loss 悪化 (≥3.57) | over-train、 epoch=30 が真の最適 | epoch=30 を canonical 維持 |

## リスク / 既知の不確実性

- **訓練時間 30 分** = 想定 +12 分でコスト微増、 cost-limit 内
- **vs baseline_v1 対戦結果**: iter1 (epoch=30) で 3/50 だったが n=50 では信頼不可、 本実験でも 50戦のみ実施。 真の改善判定には 300戦再評価が必要 (本実験のスコープ外)
- **lr scheduler 不在**: epoch=50 でも constant lr=1e-3 のまま、 over-train の可能性あり (採否しきい値の最後のシナリオ)
