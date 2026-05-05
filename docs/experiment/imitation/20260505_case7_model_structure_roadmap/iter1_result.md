# imitation/case7 — Set Transformer + cross-attention target head (iter1 結果)

> 作成日: 2026-05-05
> 関連: [`plan.md`](./plan.md) (4-iter roadmap)、 [`docs/experiment/imitation/20260504_case6_attention_backbone/result.md`](../20260504_case6_attention_backbone/result.md)
> Plot: `data/output/experiment/imitation_case7_set_transformer/case5_vs_case6_vs_case7_comparison.png` (3-way ablation 4-panel)
> 結論: **学習完走、 val_loss / val_from_acc / val_ships_acc は case6 比で改善 ✅、 ただし**
> **本 iter の主仮説「cross-attention target head で val_target_acc を +3pp」は不成立 (+0.16pp で誤差レベル)** ❌

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| case7 実装 (Set Transformer + cross-attention head) | ✅ 完了 (commit `983f09f` → format/sanity check 修正後 `1575119`) |
| 単体テスト (forward shape / mask / NaN safe / ISAB / PMA / target diversity) | ✅ 7/7 pass |
| ローカル smoke (10 ep / 2 epoch) | ✅ end-to-end 動作確認 |
| RunPod 学習 v2 (`1575119`) | ✅ **15 epoch 完走** (RTX 4090, 9 分 train, $0.65) |
| best.pt 取得 | ✅ S3 fallback で取得済 |
| **plan.md 主仮説の検証** | ❌ **rejected** (val_target_acc は case6 と誤差レベル) |
| 副次効果 | ✅ val_loss −0.030, val_from +2.1pp, val_ships +4.5pp |

## 2. 学習結果 (RunPod v2 完走、 commit 1575119)

### 2.1 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260505-090036__feature-imitation-model-structure__1575119__seed0` |
| pod_id | `q6sdv3a3wwo7rs` |
| GPU | RTX 4090 SECURE @ $0.690/h |
| image | `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204` |
| epochs | **15 (best_epoch=14)** |
| best_val_loss | **3.6211** (case6 3.6514 比 −0.0303) |
| runtime (train) | 546 秒 (~9 分) |
| 実コスト | $0.65 (preprocess 30 分 + train 9 分 + 環境構築 12 分) |

### 2.2 学習曲線 (`metrics.json`)

| epoch | train_total | val_total | val_from_acc | val_target_acc | val_ships_acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.976 | 3.839 | 0.920 | 0.359 | 0.847 |
| 5 | 3.711 | 3.695 | 0.926 | 0.397 | 0.862 |
| 10 | 3.649 | 3.657 | 0.928 | 0.407 | 0.863 |
| 13 | 3.614 | 3.629 | 0.933 | 0.418 | 0.860 |
| **14 (best)** | **3.601** | **3.621** | **0.929** | **0.421** | **0.865** |

train/val loss は最終 epoch まで monotonic に減少 → **15 epoch では capacity に対して足りない**、 案 D (規模拡大) の追求余地あり。

## 3. 3-way ablation 比較 (case5 / case6 / case7)

**完璧な ablation 条件**: 同 featurizer (PLANET_FEAT_DIM=17、 ship-prediction 6 列込み) + 同ハイパラ (epochs=15, batch=256, lr=1e-3, hidden=128) で **backbone+head のみ差分**。

| metric | case5 (GraphConv) | case6 (Attention) | **case7 (Set Transformer)** | Δ vs case6 | 評価 |
|---|---|---|---|---|---|
| **best_val_loss** | 3.7285 | 3.6514 | **3.6211** | **−0.0303** | ✅ 改善 |
| best_epoch | 8 | 12 | **14** | (last epoch) | より深い学習可能 |
| **val_from_acc (best)** | 0.8666 | 0.9076 | **0.9286** | **+0.0210** (+2.1pp) | ✅ 顕著な改善 |
| **val_target_acc (best)** | 0.3951 | 0.4190 | **0.4206** | **+0.0016** (+0.16pp) | ❌ 頭打ち継続 |
| **val_ships_acc (best)** | 0.7899 | 0.8197 | **0.8650** | **+0.0453** (+4.5pp) | ✅ 大改善 |
| runtime (sec) | 329 | 434 | **546** | +112 | コスト増 |

### 観察

- **val_from_acc** は case5 → 6 → 7 で順に改善 (+4.1pp → +2.1pp)、 attention で「どの planet から発射するか」識別能力が単調向上
- **val_ships_acc** は case6 で +3pp、 case7 で更に +4.5pp。 **case7 の Set Transformer encoder が ships head に最も効いている** (ISAB の global context を経由するため、 source planet の状況を文脈化して ships 量を decide できる)
- **val_target_acc は case6 (0.4190) → case7 (0.4206) で +0.16pp**。 Set Transformer + cross-attention head でも target diversity 頭打ちは破れず → **plan 仮説の核心は rejected**
- best_epoch=14 (= 最終 epoch) で val_loss は **まだ下がり続けている** → epochs=15 では capacity に対して訓練量不足、 epochs=20-30 + 規模拡大の余地あり

### plot

`data/output/experiment/imitation_case7_set_transformer/case5_vs_case6_vs_case7_comparison.png` (4-panel:
val_loss + val_from + val_target + val_ships)

特に val_target_acc panel (左下) で **case6 (赤) と case7 (緑) のラインが終盤でほぼ重なる** 様子が visible。

## 4. 仮説検証

plan.md (iter1 = case7 部分) の主仮説:

> **template を learnable query にした cross-attention head なら val_target_acc 0.42 → 0.45+ に押し上げ可能。**

**結果**: case7 val_target_acc = **0.4206** (case6 の 0.4190 比 +0.0016)。 +3pp threshold (0.45) には届かず、 **rejected**。

副次仮説:

> **Set Transformer encoder (ISAB stack) で val_loss を case6 比 −1% 改善**

**結果**: val_loss 3.6211 (case6 3.6514 比 **−0.83%**)、 threshold (−1%) ぎりぎり下回るが概ね成立。 val_from / val_ships はより明確に改善。

## 5. RunPod インフラ trap (本 iter で 1 件追加検出)

| 失敗 | 原因 | 修正 |
|---|---|---|
| **case7 v1 (commit b329c7f) train 起動時 reshape error** | preprocess 出力 parquet の `planet_feats` 列が一部 frame で jagged → numpy reshape 失敗。 case6 で同コードが動いていたため stochastic 性疑い | `1575119`: preprocess.done を打つ前に `np.array(...).reshape(...)` で sanity check、 失敗時は preprocess 段階で fail (train pod cost 節約) |

case7 v1 損失: $0.59、 v2 で完走 ($0.65) → iter1 計 **$1.24**。

## 6. 採否判断

- **本 iter の主仮説 (cross-attention target head で target_acc 改善) は rejected** ❌
- **副次的に val_loss / val_from / val_ships は case6 比で改善** ✅、 model.py のコード自体は問題なく動作
- 採否: **rejected by experiment outcome** (target_acc 改善が出なかったため、 4-iter roadmap の他 axis に進む価値あり)
- weights.pt は `bot/pipeline/imitation/case7/policy/weights.pt` に配置済 (`dev/runpod promote` 不要、 canonical は case1 のまま)

## 7. iter2 (case8 Pointer Network) への引継ぎ

case7 で **target_head の構造変更 (cross-attention)** だけでは target_acc が伸びないと判明 → 次は **head 設計の別軸 = autoregressive (Pointer Network)** に移る。

予想される効果分布 (4-iter roadmap の比較):

| iter | 構造 | val_target_acc 予想 | 根拠 |
|---|---|---|---|
| iter1 (case7) | Set Transformer + cross-attn head | 0.42 (実測 0.4206) ❌ | head 構造変更だけでは伸びず |
| iter2 (case8) | Hierarchical Pointer Net (autoregressive) | 0.43-0.45? | from→target dependency を明示学習、 target diversity 増加期待 |
| iter3 (case9) | E(2)-EGNN | 0.42-0.43? | 学習効率 ↑、 ただし target axis 直接の改善は見込み薄 |
| iter4 (case10) | hidden 256, layers 5 | 0.43-0.45? | case7 で「epoch 14 が best = capacity 不足」 = 規模拡大余地あり |

iter2 (case8) は当初 plan の優先度通り進める。 implementation 時は case7 の **Set Transformer encoder を流用** + decoder のみ Pointer Net に置換する形が実装コスト最小。

## 8. 累計コスト (case7 iter1)

| run | commit | コスト | 結果 |
|---|---|---|---|
| v1 | `b329c7f` | $0.59 | 65_train_failed (parquet reshape error) |
| v2 | `1575119` | $0.65 | ✅ 15 epoch 完走 |
| **合計** | — | **$1.24** | iter1 完走 |

## 9. 補助評価: vs baseline_v1 50戦 (sanity)

```
episodes:    50
wins:         0
losses:      50
draws:        0
win_rate:    0.0%   (95% Wilson CI: 0.0% – 7.1%)
challenger:  il_v7
baseline:    baseline_v1
seed:        0..49
```

→ **case6 と完全同じ 0/50**。 val 指標は case6 比で改善 (val_loss −0.030, val_from +2.1pp, val_ships +4.5pp) しているのに、 対戦勝率は変化なし → **imitation 系の「val 良好でも対戦で全敗」のパターンを再現** (memory `project_imitation_case1_2026_04_19`)。

backbone+head 設計の改修だけでは vs baseline_v1 を破れない。 plan.md の「補助 sanity」位置付け通り、 採否判定には使わない (n=50 では強い主張不可)。
