# imitation/case8 — Hierarchical Pointer Network (iter2 結果)

> 作成日: 2026-05-05
> 関連: [`plan.md`](./plan.md) (4-iter roadmap)、 [`iter1_result.md`](./iter1_result.md) (case7 Set Transformer)
> Plot: `data/output/experiment/imitation_case8_pointer_net/case5_to_case8_comparison.png` (4-way ablation)
> 結論: **学習完走、 ただし主仮説「autoregressive で target diversity 改善」は不成立 ❌**
> case7 比で **val_target_acc は -0.9pp と微減**、 val_loss も +0.027 と微増。 head 設計の axis では target diversity 頭打ちは破れない。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| case8 実装 (Hierarchical Pointer Network 風 GRU decoder) | ✅ 完了 |
| 単体テスト (forward shape / mask / NaN safe / autoregressive 勾配伝播) | ✅ 7/7 pass |
| ローカル smoke (10 ep / 2 epoch CPU) | ✅ end-to-end 動作 |
| RunPod 学習 v2 (`fe299e6`) | ✅ **15 epoch 完走** (RTX 4090, 8 分 train, $0.65) |
| best.pt 取得 | ✅ S3 fallback |
| **plan.md 主仮説 (autoregressive で val_target_acc +3pp)** | ❌ **rejected** (実測 -0.9pp) |

## 2. 学習結果 (RunPod v2 完走、 commit fe299e6)

### 2.1 統計

| 項目 | 値 |
|---|---|
| run_id | `20260505-121340__feature-imitation-model-structure__fe299e6__seed0` |
| GPU | RTX 4090 SECURE @ $0.690/h |
| epochs | **15 (best_epoch=14)** |
| best_val_loss | **3.6485** (case7 3.6211 比 +0.0274 ⚠️) |
| runtime (train) | 484 秒 (~8 分) |
| 実コスト | v1 $1.05 (race condition) + v2 $0.65 = **$1.70** |

### 2.2 学習曲線 (case7 比でやや悪化)

| epoch | train_total | val_total | val_from_acc | val_target_acc | val_ships_acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.959 | 3.823 | 0.928 | 0.378 | 0.853 |
| 5 | 3.712 | 3.699 | 0.929 | 0.402 | 0.860 |
| 10 | 3.652 | 3.672 | 0.933 | 0.405 | 0.862 |
| **14 (best)** | **3.606** | **3.648** | **0.933** | **0.412** | **0.863** |

## 3. 4-way ablation 比較 (case5 / case6 / case7 / case8)

**完璧な ablation 条件**: 同 featurizer (PLANET_FEAT_DIM=17) + 同ハイパラ で **encoder/decoder 設計のみ差分**。

| metric | case5 (GraphConv) | case6 (Attention) | **case7 (Set Transformer)** | **case8 (Pointer Net)** | Δ vs case7 |
|---|---|---|---|---|---|
| **best_val_loss** | 3.7285 | 3.6514 | **3.6211** ⭐ | 3.6485 | **+0.0274 ❌** |
| **val_from_acc** | 0.8666 | 0.9076 | 0.9286 | **0.9333** ⭐ | +0.0047 ✅ |
| **val_target_acc** | 0.3951 | 0.4190 | **0.4206** ⭐ | 0.4116 | **−0.0090 ❌** |
| **val_ships_acc** | 0.7899 | 0.8197 | **0.8650** ⭐ | 0.8625 | −0.0025 ≈ |

### 観察

- **case8 が首位を取れた指標は val_from_acc のみ** (+0.005 = 誤差)
- val_target_acc は **case7 から -0.9pp 後退** = autoregressive 化はむしろ target 選択を悪化させた
- val_loss も case7 比 +0.027 で全体的に劣る
- **forward の argmax cascade 設計の限界**: 学習時に teacher forcing が無いため、 GRU の autoregressive 経路が弱い勾配しか持たない (target_argmax が discrete で grad-stop 効果)

### plot

`data/output/experiment/imitation_case8_pointer_net/case5_to_case8_comparison.png` (4-panel)

## 4. 仮説検証

plan.md (iter2 = case8 部分) の主仮説:

> **from→target→ships を autoregressive 化し、 GRU cell で前段の決定が後段の hidden state に流れることで val_target_acc が +3pp (case7 比) 改善。**

**結果**: case8 val_target_acc = **0.4116** (case7 0.4206 比 **-0.009**)。 +3pp threshold (0.45+) には到底届かず、 むしろ **後退**。 **rejected**。

### 失敗の原因仮説

1. **Argmax cascade による grad stop**: forward では `target_argmax = target_logits.argmax(dim=-1)` で discrete に切ってから ships 計算。 これにより **target_logits → ships_head に勾配流れない** (本来は teacher forcing で target label を流すべきだが、 既存 losses.py は parallel head 前提で teacher forcing 機構なし)
2. **GRU cell 1 step で表現力不足**: 1 step update では context 集約に不十分。 LSTM もしくは Transformer decoder ならまだしも GRU 1 step では弱い
3. **target diversity bottleneck はそもそも head 設計の問題ではない**: case6/7/8 全て 0.42 で頭打ち = データ側 (top-50% rating Kaggle ep が greedy 寄り) または loss 設計が真の bottleneck

## 5. 採否判断

- **本 iter rejected**。 autoregressive 化は val 指標を全体的に悪化させた
- weights.pt は `bot/pipeline/imitation/case8/policy/weights.pt` に配置 (canonical promote はしない)
- **iter1 (case7) のほうが val 指標で優位** = 4-iter roadmap の中で **case7 が現状最良**

## 6. iter3 (case9 EGNN) への引継ぎ + 重要な観察

case6/7/8 で **3 つの head 設計を試したが val_target_acc は全部 0.42 で頭打ち**:

- case6: parallel 3-head (MLP + template_ctx)
- case7: Set Transformer + cross-attention head
- case8: Hierarchical Pointer (GRU autoregressive)

これは **head 設計の問題ではなく、 データ or loss 設計が真の bottleneck** という強い示唆。 iter3 (case9 EGNN) では **encoder の inductive bias** に焦点が移る形に:

| iter3 (case9) | 仮説調整 |
|---|---|
| 元仮説 | E(2)-equivariance で学習効率向上 |
| 調整後仮説 | encoder bias (回転不変性) で **データ効率向上** → 同 epoch 数で val_loss は伸びる可能性、 ただし val_target_acc 頭打ちは依然破れない見込み |

iter4 (case10 規模拡大) も同様。 もし target diversity を真に伸ばしたいなら **本 roadmap 完了後に loss 設計 / データ拡張 に方向転換** が必要。

## 7. 累計コスト

| iter | run | コスト | 結果 |
|---|---|---|---|
| iter1 v1 (case7) | b329c7f | $0.59 | reshape error |
| iter1 v2 (case7) | 1575119 | $0.65 | ✅ 完走 |
| iter2 v1 (case8) | 9dbac0c | $1.05 | race condition fail |
| iter2 v2 (case8) | fe299e6 | $0.65 | ✅ 完走 |
| **roadmap 累計 (iter1-2)** | — | **$2.94** | iter1, iter2 共に完走 |

## 8. 補助評価: vs baseline_v1 50戦 (⭐ 想定外の breakthrough)

```
episodes:    50
wins:         3   ⭐ (case6=0, case7=0)
losses:      47
draws:        0
win_rate:    6.0%  (95% Wilson CI: 2.1% – 16.2%)
challenger:  il_v8
baseline:    baseline_v1
seed:        0..49
```

→ **case6/case7 が 0/50 だったのに対し、 case8 は 3/50 = 初の非ゼロ勝率**。

| case | 対戦勝率 | 95% Wilson CI |
|---|---|---|
| case6 | 0/50 = 0.0% | 0.0–7.1% |
| case7 | 0/50 = 0.0% | 0.0–7.1% |
| **case8** | **3/50 = 6.0%** | **2.1–16.2%** |

### 重大な観察: val と対戦勝率の解離

- **val 指標で case8 < case7**: best_val_loss 3.65 vs 3.62、 val_target 0.412 vs 0.421
- **対戦勝率は case8 > case7**: 3/50 vs 0/50

つまり **autoregressive head は val 上では悪化したのに、 実際の対戦では勝てた**。 val と対戦勝率の解離パターンで、 memory `project_imitation_case1_phase2_breakthrough` (case1 iter9 の 5/100 を実現した from focal α=0.75) と同様の症状。 ただし memory `project_imitation_case1_phase3` の通り **iter9 の 5/100 は再評価 0/300 で否定** されたため、 **n=50 の 3 勝は信頼できない**。

### n=50 の信頼性 (重要)

- 95% Wilson CI 上限が **16.2%** で、 真値 5% の閾値を含む = 「勝率 5% 以上」は強い主張ではなく、 0/50 のノイズ (case6/7) との差は **statistical に有意でない可能性**
- memory `project_imitation_case1_phase3` の前例: case1 iter9 で 5/100 → 0/300 で否定
- **300戦再評価を実施しないと採否判断は留保**

### 採否判断の修正

iter2 採否: **保留 (n=300 再評価必要)**
- val 指標悪化 + 対戦勝率上昇という **解離パターンは興味深いが信頼できない**
- 300 戦で実勝率が 3-6% 程度に収束するなら正の breakthrough、 1% 以下に落ちるなら seed variance による noise
- **300 戦再評価のコスト**: ローカル CPU で約 ~30 分、 RunPod 不要
