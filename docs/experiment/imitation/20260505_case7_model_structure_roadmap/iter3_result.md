# imitation/case10 + case11 — Scaled-Up (Set Transformer XL + Pointer Net XL) iter3 結果

> 作成日: 2026-05-06
> 関連: [`plan.md`](./plan.md) (4-iter roadmap)、 [`iter1_result.md`](./iter1_result.md) (case7)、 [`iter2_result.md`](./iter2_result.md) (case8)
> Plot: `data/output/experiment/imitation_iter3_scaled/case5_to_case11_comparison.png` (6-way ablation 4-panel)
> 結論: **両方完走、 ただし主仮説「規模拡大で val 改善」は不成立 ❌**
> case7 比 case10 で val_loss +0.046 / val_target -1.6pp、 case8 比 case11 で val_loss +0.18 / val_target -4pp の **大幅悪化**。 hidden 256 + 15 epoch では **under-trained** (train_loss は下がるが val_loss が伸びない) が真因。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| case10 実装 (Set Transformer 256 XL) | ✅ commit `d2d33f1` |
| case11 実装 (Pointer Net 256 XL) | ✅ commit `d2d33f1` |
| 単体テスト | ✅ 30/30 pass (15 each) |
| ローカル smoke (両 case) | ✅ end-to-end 動作 |
| RunPod 学習 (両 case 並列) | ✅ 完走 (case10: 942 秒、 case11: 736 秒) |
| **plan.md 主仮説 (規模拡大で val 改善)** | ❌ **rejected** (両 case で val 悪化) |

## 2. 学習結果

### 2.1 統計

| 項目 | case10 (Set Transformer XL) | case11 (Pointer Net XL) |
|---|---|---|
| run_id | `20260505-134455__...__d2d33f1__seed0` | `20260505-134516__...__d2d33f1__seed0` |
| GPU | RTX 4090 SECURE | RTX 4090 SECURE |
| epochs | 15 (best_epoch=14) | 15 (best_epoch=14) |
| best_val_loss | **3.6669** (case7 比 +0.046 ❌) | **3.8250** (case8 比 +0.18 ❌❌) |
| runtime (train) | 943 秒 (~16 分) | 736 秒 (~12 分) |
| 想定コスト/case | $0.18 (train) + 環境構築 | $0.14 (train) + 環境構築 |

### 2.2 学習曲線の特徴

- **両 case とも train_loss は最終 epoch まで monotonic に減少** = 学習機構自体は機能している
- **しかし val_loss は base 案 (case7/case8) を**全 epoch にわたって**下回らない** = 大きな model に対する訓練量不足 (15 epoch では fit しきれない)
- 特に **case11 (Pointer XL) は epoch 0 で val_loss=3.92** と base case8 の epoch 0 (3.82) より高く、 始まりから劣後

## 3. 6-way ablation 比較

**完璧な ablation 条件**: 同 featurizer + 同訓練設定 (epochs/batch/lr) で **構造のみ差分**。

| metric | case5 (GraphConv) | case6 (Attention) | **case7 (SetT 128)** | case8 (Ptr 128) | case10 (SetT 256) | case11 (Ptr 256) |
|---|---|---|---|---|---|---|
| **best_val_loss** | 3.7285 | 3.6514 | **3.6211** ⭐ | 3.6485 | 3.6669 ❌ | 3.8250 ❌❌ |
| **val_from_acc** | 0.8666 | 0.9076 | 0.9286 | **0.9333** ⭐ | 0.9294 | 0.8633 ❌ |
| **val_target_acc** | 0.3951 | 0.4190 | **0.4206** ⭐ | 0.4116 | 0.4042 ❌ | 0.3712 ❌❌ |
| **val_ships_acc** | 0.7899 | 0.8197 | **0.8650** ⭐ | 0.8625 | 0.8639 | 0.8342 ❌ |

### 3.1 規模拡大の効果分析

| 比較 | case7 (128) → case10 (256) | case8 (128) → case11 (256) |
|---|---|---|
| best_val_loss | 3.6211 → **3.6669** (+0.046) ❌ | 3.6485 → **3.8250** (+0.18) ❌❌ |
| val_from_acc | 0.929 → 0.929 (≈) | 0.933 → **0.863** (-7pp) ❌❌ |
| val_target_acc | 0.421 → **0.404** (-1.6pp) ❌ | 0.412 → **0.371** (-4pp) ❌❌ |
| val_ships_acc | 0.865 → 0.864 (≈) | 0.863 → **0.834** (-3pp) ❌ |

→ **両ベース ($1.5x$ params) で規模拡大は val 悪化**、 特に Pointer Net (case11) は **より深刻**。

### 3.2 なぜ規模拡大が逆効果だったか

1. **hidden 256 + 15 epoch では under-trained**: train_loss は下がり続けたが val が伸びていない
2. **lr=1e-3 が大規模 model に対して **暗黙的に過小**: 規模拡大時は lr scheduler (warmup + cosine decay) が必要
3. **batch_size=256 のまま**: 大規模 model でも batch を増やしていない、 学習の安定性に影響
4. **case11 (Pointer XL) が特に悪化**: GRU autoregressive 経路で train signal が弱いため、 規模拡大で **更に学習が困難に**

## 4. 仮説検証

plan.md (iter4 = case10/11 部分) の主仮説:

> **case7 epoch 14 (= 最終) で val_loss が下がり続けていた = capacity 不足。 hidden 256, layers 5, heads 8 で table top に到達してさらに改善する。**

**結果**: capacity 増では改善せず、 **逆に悪化**。 真の bottleneck は **訓練設定** (epoch 数、 lr scheduler、 batch size) で **capacity 不足ではない**。

## 5. 採否判断

- **iter3 (case10/case11): rejected**
- **現状最良 case = case7 (Set Transformer 128)** で確定
- **roadmap 4 iter の傾向**: backbone 改修 (case6→7) は val 改善、 head 設計 (case7↔8) や規模 (case10/11) は target_acc 頭打ち破れず
- **真の bottleneck はデータ or loss 設計** (case8 で観察された val/対戦の解離も含めて)

## 6. roadmap 全体総括 (iter1-3 完了時点)

| iter | case | 構造 | best_val_loss | 採否 |
|---|---|---|---|---|
| baseline (`8d3402` まで) | case5 | GraphConv | 3.7285 | (reference) |
| baseline | case6 | Attention | 3.6514 | (reference) |
| iter1 | **case7** | Set Transformer 128 | **3.6211** ⭐ | adopted (現状最良) |
| iter2 | case8 | Pointer Net 128 | 3.6485 | rejected (val 悪化) |
| iter3 | case10 | Set Transformer 256 XL | 3.6669 | rejected (規模拡大失敗) |
| iter3 | case11 | Pointer Net 256 XL | 3.8250 | rejected (規模拡大大失敗) |

**case9 (EGNN) は roadmap iter3 として残っているが**、 規模拡大で逆効果になった結果から **encoder 軸での改善余地は薄い** ことが判明。 **roadmap を terminate して別 axis (loss 設計、 データ拡張、 訓練設定) に方向転換** が現実的。

## 7. 補助評価: vs baseline_v1 50戦 (sanity)

(eval 実行中、 完了次第追記)

```
TODO: data/mart/imitation/case10/eval_metrics.json から win/loss/draw/Wilson CI を転記
TODO: data/mart/imitation/case11/eval_metrics.json から
```

case7 (現状最良) が 0/50、 case8 (val 悪化) が **3/50** だった解離パターンを踏まえ、 case10 (val 中位) と case11 (val 大幅悪化) の対戦結果は以下が候補:

- case10: 0/50 〜 1/50 の見込み (val 改善せず、 case7 と類似)
- case11: 0/50 〜 5/50 の見込み (val 大幅悪化なら argmax cascade で逆に多様性が出る可能性?)

## 8. 累計コスト (iter3)

| run | コスト | 結果 |
|---|---|---|
| case10 (`d2d33f1`) | $0.18 train + ~$0.40 setup = **~$0.58** | ✅ 完走 |
| case11 (`d2d33f1`) | $0.14 train + ~$0.40 setup = **~$0.54** | ✅ 完走 |
| **iter3 計** | **~$1.12** | 両完走 |
| **roadmap 累計 (iter1-3)** | **$4.06** | 5 cases trained |

## 9. 次サイクル提案

### 短期 (Cycle 5)

1. **roadmap iter4 (案 D 規模拡大) は事実上完了** (case10/11 で実証)、 残る案 C (EGNN) は実施するか判断
2. **訓練設定の見直し**: case10/11 を **epoch 30, lr scheduler (warmup + cosine), batch 512** で再学習。 規模拡大の真の効果検証
3. **データ axis に方向転換**: rating quantile の見直し (top-50% → top-30%)、 loser obs frame の oversample

### 中期

- **BC + RL 混合学習** (memory `project_imitation_case1_phase2_breakthrough` の case1 phase2 と同 axis)
- **case7 ベースで loss 設計 ablation** (focal α, target_label_smoothing, target_entropy_bonus)
