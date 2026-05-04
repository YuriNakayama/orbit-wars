# Imitation Case6 — Graph Attention U-Net Backbone

case5 (ship-prediction featurizer) をコピーし、Graph U-Net backbone の
**GraphConv (degree-normalized mean) を Graph Attention (multi-head, edge feat)** に置き換えた
imitation case。featurizer は case5 と完全同一 (PLANET_FEAT_DIM=17)。

仮説: attention 化で「どの planet が重要か」をモデルに明示学習させると、target/ships head の
スコアリング精度が向上し、対戦勝率を押し上げる。

詳細: [`docs/experiment/imitation/20260504_case6_attention_backbone/plan.md`](../../../../docs/experiment/imitation/20260504_case6_attention_backbone/plan.md)

## ディレクトリ構成

```
pipeline/imitation/case6/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # obs → torch.Tensor (case5 と同一、17 dim)
│   ├── timeline.py          # ship-prediction (case5 と同一)
│   ├── model.py             # ★ GraphAttentionUNetPolicy (case6 で唯一の差分)
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── templates.py         # target template 定義
│   └── weights.pt           # 学習済み重み (DVC 管理、git untracked)
├── configs/
│   └── il_case6.yaml        # case5 config + attn_heads=4
├── training/                # 開発用 (.submitignore)
└── evaluation/              # 開発用 (.submitignore)
```

## アーキテクチャ差分 (vs case5)

| 層 | case5 (GraphConv) | case6 (GraphAttention) |
|----|-------------------|-------------------------|
| 集約 | `mean(neighbours)` (uniform) | multi-head attention (4 heads) |
| edge feature | 未使用 | `_pairwise_geometry` 6 dim を attention bias として注入 |
| パラメータ数 | ~165k | ~250-300k |

`GraphAttention.forward(h, adj, mask, edge_feats)` の挙動:

```
e_ij = LeakyReLU( (W_q h_i) · (W_k h_j) / √D + W_e e_ij )
α_ij = softmax_j(e_ij over j ∈ N(i))
h'_i = ReLU( W_o · concat_h(Σ_j α_ij W_v h_j) )
```

`top_idx` で pooling した後の階層では、edge feature も同じ index で gather して縮約する
(`_gather_pair_feats`)。これにより各 attention 層が物理的に意味のある (dx, dy, dist, ...) を持つ。

## 手順

### DVC 経由

```bash
# case6 は case5 と featurizer 同一なので preprocess は case5 stage で済ませる。
uv run --directory bot dvc repro preprocess_imitation_case5
uv run --directory bot dvc repro train_imitation_case6
```

### 直接実行

```bash
cd bot

# 1) 前処理 (case5 の preprocess で兼用、parquet は data/mart/imitation/case5/ に出る)
uv run python -m pipeline.imitation.case5.training.preprocess --config pipeline/imitation/case5/configs/il_case5.yaml

# 2) 学習 (案件は data/mart/imitation/case5/{train,val}.parquet を読む)
uv run python -m pipeline.imitation.case6.training.train --config pipeline/imitation/case6/configs/il_case6.yaml

# 3) ローカル評価 (vs baseline_v1)
uv run python -m pipeline.imitation.case6.evaluation.eval_vs_baseline --episodes 50 --seed 0
```
