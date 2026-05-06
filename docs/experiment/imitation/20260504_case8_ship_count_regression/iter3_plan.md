# imitation/case8 — iter3: focal loss で cand head の label imbalance に対処

> 作成日: 2026-05-05
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) (ship_head 設計)
> - [`iter1_result.md`](./iter1_result.md) (1/50 = 2.0%、cand head 振動)
> - [`iter2_plan.md`](./iter2_plan.md), [`iter2_result.md`](./iter2_result.md) (cosine LR + early stop で 0/50 = 0.0%、val metric と win rate の相関崩壊)
> - `bot/pipeline/imitation/case8/training/losses.py` (現行 CE + class_weight)
>
> スコープ: cand head の cross-entropy を **focal loss (α, γ)** に置換。class_weight は維持。ship_head / scheduler / early stop は iter2 のまま不変。

## 仮説 (Hypothesis)

iter1/iter2 の cand head は val_cand_acc が epoch ごとに 0.01-0.37 で **逆相関的に振動** (no-op 寄り epoch と fire 寄り epoch が交互)。

iter2 で cosine + warmup を追加しても振動は消えず、win_rate は 2.0% → 0.0% に悪化。
val_cand_fire_acc を best metric にしても win_rate と相関しない。

**根本原因の仮説**: cand head の training data は **`cand_slot=0` (no-op) ラベルが圧倒的多数** (preprocess 時に my_planet がないが fire しない場合は noop label を付ける)。CE + inverse-freq class_weight で「present class 平均=1.0」に正規化しても、majority class の easy examples (= 簡単な noop) が累積 loss を支配し、fire 例の hard examples の勾配が埋もれる。

**Focal Loss** (Lin et al., 2017, "Focal Loss for Dense Object Detection"):
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- `(1 - p_t)^γ` が **easy examples (p_t 高い、確信のある正解) の loss を down-weighting**
- `α_t` (per-class) で更に minority class (fire slots 1-7) を up-weighting
- γ=2.0 が標準値 (object detection で実証)、α=0.25 (minority 寄り)

これを cand_loss に適用すれば **easy noop の勾配を抑え、hard fire 例の勾配比率を上げる** ことで cand head が fire 判定を真に学べる可能性が高い。**ship_head は不変** (iter1/iter2 で機能確認済み)、scheduler は iter2 のまま (機構自体は OK)。

期待: vs baseline_v1 50 戦で **>5% (3/50 以上)** に到達、cand_acc 振動も振幅縮小。

## 既存コードの現状 (iter1/iter2 から継承)

- `policy/model.py` / `types.py` / `decoder.py` / `featurizer.py` / `candidates.py` / `agent.py`: 不変
- `policy/geometry.py`: 不変
- `training/preprocess.py`: 並列化済み、不変
- `training/dataset.py`: ship_label_per_src 対応済み、不変
- `training/train.py`: scheduler / early stop / debug stamp、本 iter で **`compute_loss` 呼び出しの引数追加なし** (focal loss は losses.py 内で透明に切替)
- `training/losses.py`: 現行 `cross_entropy(weight=class_weights, label_smoothing=...)` → 本 iter で focal 切替

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|---|---|
| `bot/pipeline/imitation/case8/training/losses.py` | `LossWeights` に `cand_loss_type: str = "ce"` / `focal_alpha: float = 0.25` / `focal_gamma: float = 2.0` を追加。`compute_loss` で `cand_loss_type=="focal"` なら自前 focal loss を呼ぶ (label_smoothing は focal と非整合なので focal 時は無視 or warn) |
| `bot/pipeline/imitation/case8/configs/il_case8.yaml` | `loss_weights.cand_loss_type: focal` / `focal_alpha: 0.25` / `focal_gamma: 2.0` を追加。`label_smoothing: 0.0` に変更 (focal と併用しない) |
| `bot/pipeline/imitation/case8/training/train.py` | `LossWeights(...)` の constructor に新フィールドを渡すよう更新 |
| `bot/tests/pipeline/imitation/case8/test_focal_loss.py` (新規) | (1) gamma=0, alpha=均等 → CE と一致。(2) class imbalance で focal < CE。(3) 勾配が finite。(4) backward が通る。(5) all-zero target で loss=0 |

### 変更なし

- ship_head / SmoothL1 ship_loss / λ_ship — iter1/iter2 で機能確認済み、不変
- scheduler (cosine warmup) / early stop / best_metric — iter2 で機構動作確認、不変
- backbone (Graph U-Net) / candidate set / featurizer — 全 iter 共通

### Hyperparameters

| Knob | iter2 | iter3 |
|---|---|---|
| cand loss | CE + label_smoothing=0.05 + class_weights | **focal (α=0.25, γ=2.0) + class_weights** |
| label_smoothing | 0.05 | **0.0** (focal と併用しない) |
| class_weight | inverse-freq (present mean=1.0) | **同じ inverse-freq を維持** (focal の α と直交、両方適用) |
| ship loss | SmoothL1, λ=1.0 | 不変 |
| scheduler | cosine_warmup (T_max=30, warmup=2) | 不変 |
| epochs | 30 | 30 |
| early_stop | val_cand_fire_acc, patience=5, max | 不変 |
| best_metric | val_cand_fire_acc, max | 不変 |

### Focal loss 実装の数式

```python
def focal_cross_entropy(
    logits: Tensor,        # (N, C) raw logits
    targets: Tensor,       # (N,) class indices
    alpha: Tensor | float, # per-class or scalar
    gamma: float,
    weight: Tensor | None, # class_weight (multiplicative, 別系)
) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)            # (N, C)
    log_pt = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    pt = log_pt.exp()                                     # (N,)
    focal_weight = (1 - pt) ** gamma                      # (N,)
    if isinstance(alpha, Tensor):
        alpha_t = alpha.to(logits.device).gather(-1, targets)
    else:
        alpha_t = alpha
    loss_per_sample = -alpha_t * focal_weight * log_pt    # (N,)
    if weight is not None:
        weight_t = weight.to(logits.device).gather(-1, targets)
        loss_per_sample = loss_per_sample * weight_t
    return loss_per_sample.mean()
```

CE との互換性: γ=0 かつ α=1.0 (定数) なら `focal_weight = 1`, `alpha_t = 1` で `-log_pt` = CE と一致。

## 実装ステップ (Implementation outline)

1. `losses.py`: `_focal_cross_entropy(logits, targets, alpha, gamma, weight)` を追加 (上式)。`compute_loss` で `weights.cand_loss_type == "focal"` なら focal を呼ぶ
2. `LossWeights` dataclass に 3 fields を追加 (default は CE)
3. `train.py`: `LossWeights(...)` constructor に `cand_loss_type` / `focal_alpha` / `focal_gamma` を読んで渡す
4. `il_case8.yaml`: `loss_weights.cand_loss_type: focal` + `focal_alpha: 0.25` + `focal_gamma: 2.0` + `label_smoothing: 0.0` に変更
5. `test_focal_loss.py` 新規: 5 つの sanity check
6. `dev/test-bot` で format / lint / mypy / pytest 全通過確認
7. RunPod launch、commit + push、cron 監視
8. `dev/runpod pull --from s3` → 50 戦評価 → iter3_result.md

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case8 -x`
- **リモート**:
  - `dev/runpod train <new_sha> --case case8`
  - 想定所要時間: container 起動 16min (前 iter と同等) + dvc pull 5-7min + train 30 epoch ≒ ~18min + post ≒ **~50 min total**
  - 想定コスト: ~$0.55 (RTX 4090 SECURE $0.69/h × 0.8h)
- **評価**:
  - 対戦相手: `baseline_v1`
  - エピソード数: **50 戦 (sanity)** → win_rate >5% (3/50 以上) なら **300 戦に拡張**
  - 主要メトリクス: vs baseline_v1 win_rate
  - 採否しきい値:
    - 50 戦で **>5%** → 300 戦昇格
    - 300 戦で **>7% (Wilson 95% CI lower bound > 5%)** → adopted、`dev/runpod promote` を user 確認後実行
    - 50 戦で **≤2%** → focal loss 単独では不十分、別アプローチ (oversample / hard negative mining) 検討

## リスク / 注意点

1. **focal loss + class_weight 二重適用**: focal の α と class_weights は **直交** (α は per-class scaling、class_weights は inverse-freq) で原理的に併用可能だが、「fire 寄り」を二重に強調しすぎて cand head が fire を全肯定する predictor に振れるリスク。50 戦で 0/50 のままなら α/γ sweep が必要
2. **label_smoothing と focal の併用は理論上 ill-defined**: focal は target に確信を要求、label_smoothing は target を崩す → 矛盾。本 iter は label_smoothing=0.0 に下げる
3. **gamma=2.0 は object detection の標準値**: text classification では γ=1.0 / 0.5 が良い場合あり。本 iter は標準値で先行、結果に応じて sweep
4. **noop は majority だが意味のあるラベル** (本当に fire しない方が良い場面が多い): focal で noop を down-weight しすぎると逆に noop 判定が壊れる可能性 → val_cand_noop_acc が 0 に張り付くなら対症療法として α を緩める

## 次 iter 候補 (本 plan の範囲外)

- iter4: ship head λ_ship sweep (0.5 / 1.0 / 2.0)
- iter5: data 側の oversample (fire / noop 比率を 1:1 に強制)
- iter6: cand head の構造変更 (cand_score MLP の hidden を 128 → 256 / dropout 追加)
- iter7: focal loss + label-distribution-aware margin (LDAM) の組み合わせ
