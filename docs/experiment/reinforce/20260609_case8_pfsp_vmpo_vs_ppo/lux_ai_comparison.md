# Lux AI S1 優勝解法との比較 (2026-06-16)

> 問い: モデルサイズ・手法を Lux AI Season 1 優勝 (Toe Brigade / Isaiah Pressman) と比べると?
> 出典: 優勝 repo (github.com/IsaiahPressman/Kaggle_Lux_AI_2021) を直接 fetch。

## 規模・手法の対比

| 項目 | **case8 現行** | **Lux S1 優勝** |
|---|---|---|
| アーキ | Set Transformer (ISAB×4) + per-planet pointer | **fully-conv ResNet 24 blocks** + SE |
| 幅 | hidden=192 | 128ch 5x5 conv |
| **params** | **3.15M** | **~20M** (約6倍) |
| action 空間 | per-planet pointer (~12-48 planet) | GridNet (全タイル同時) |
| algo | V-MPO | **IMPALA + UPGO + TD(λ)** |
| 教師正則化 | (未使用) | **frozen teacher の KL-divergence 正則化** |
| reward | shaping 常時 | **最初20M step shaped → sparse へ移行** |
| **curriculum** | T0/handicap (相手弱化) | **8block(shaped)→16→24block(sparse) と NN を段階拡大、小NNを教師に** |
| 計算量 | 1run ~3M transition、数run | **個人2GPUで数ヶ月の夜間学習** |

## 重要な示唆 (3点)

### ① モデルサイズ: Lux は ~20M、case8 は 3.15M (約6倍差)
- ただし Lux は **GridNet (盤面全タイルの同時行動)** で意思決定空間が桁違いに大きく、
  大容量が必要だった。Orbit Wars の per-planet pointer (planet ~12-48 の離散選択) は
  遥かに小さく、3.15M でも full 0.83 を学習できている (capacity 十分の直接証拠は
  model_size_analysis.md)。
- **とはいえ「6倍差」は将来の容量A/Bの参考値**。RL健全性修正後に伸び悩めば hidden
  192→256 (5.6M) を試す根拠になる (Lux ほどの 20M は action 空間的に過大)。

### ② Lux 優勝の核心3つが case8 に欠けている (容量より重要)
1. **frozen teacher の KL 正則化** — case8 は kl_beta 配線済だが**未使用**。Lux は
   「小さい学習済NNを教師にKL正則化」で大NNを安定学習。AlphaStar と同じ。
2. **shaped→sparse の reward 移行** — case8 は shaping 常時で「易→難の段階移行」が無い。
3. **NN を段階拡大する curriculum** — 小NN(shaped)で土台→大NN(sparse)へ。case8 の
   curriculum は「相手を弱める」方向で、Lux の「**問題/容量を段階的に上げる**」とは別物。

### ③ Lux も「shaped で土台 → sparse」= sparse 単独では学べない
Lux 優勝ですら **最初20M step は shaped reward** で土台を作り、その後 sparse に移行。
これは「sparse win-loss 単独では RL は学べない」という case8 の zero-variance 診断
(rl_failure_rootcause.md) と完全に整合。**優勝者ですら sparse を直接は攻略していない。**

## 結論

- **モデルサイズ (3.15M vs 20M)**: action 空間の差を考えれば現行で妥当。容量は現時点の
  bottleneck でない (full 0.83 学習が証拠)。ただし RL健全性修正後の容量A/Bの上限目安として
  ~5.6M (hidden256) を持っておく。Lux の 20M は GridNet 固有で本タスクには過大。
- **手法の差が本質**: Lux 優勝の勝因は size でなく (a) frozen-teacher KL 正則化、
  (b) shaped→sparse の reward 移行、(c) 段階的 capacity curriculum。case8 はこの3つを
  欠く。特に (a) KL正則化 と (b) shaped→sparse は、case8 の degenerate-batch 自壊
  (rl_failure_rootcause.md) を**直接緩和する**: KL anchor が policy の暴走 (entropy崩壊)
  を抑え、shaped が全敗バッチに非ゼロ分散の信号を与える。

→ **次手の優先順位を更新**: RL健全性修正 (A degenerate guard / B adv-std下限 / C no_op_bias)
に加え、**Lux 流の (a) frozen-teacher KL正則化 + (b) shaped→sparse 移行** を組み込むのが
最も実証された道。size 変更は後回し (必要なら 192→256 の最小A/B)。

## Sources
- [Isaiah Pressman, Kaggle_Lux_AI_2021 repo](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) — 24-block ResNet, 128ch, ~20M params, IMPALA+UPGO+TD(λ), frozen-teacher KL, 20M-step shaped→sparse, 8→16→24 block capacity curriculum
- [Lux AI S1 Grand Finale (top solutions)](https://www.toolify.ai/ai-news/lux-ai-season-1-grand-finale-unveiling-the-top-solutions-and-season-2-preview-2386553) — Toe Brigade 優勝、上位は RL
- 関連: model_size_analysis.md, rl_failure_rootcause.md, competition_solutions_research.md
