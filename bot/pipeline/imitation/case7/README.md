# Imitation Case7 — Set Transformer Backbone

case5 (ship-prediction featurizer, 17 × 6) をベースに、case6 で導入した Graph Attention U-Net を
**Set Transformer (Lee et al. 2019, arXiv:1810.00825)** に置き換えた imitation case。
featurizer は case5/case6 と完全同一。

仮説: kNN グラフを廃止し、ISAB の attention で feature similarity に基づく
集約を学習させると、planet 配置に依存しない汎化性能が向上する。

詳細: [`docs/experiment/imitation/20260505_case7_model_structure_roadmap/plan.md`](../../../../docs/experiment/imitation/20260505_case7_model_structure_roadmap/plan.md)

## アーキテクチャ差分 (vs case6)

| 層 | case6 (Graph Attention U-Net) | case7 (Set Transformer) |
|----|-------------------------------|--------------------------|
| グラフ構造 | kNN k=8 + edge features | **不要** (attention over feature similarity) |
| Encoder | 3 levels GraphAttention + TopK pool | 3 stacked **ISAB** (Induced Set Attention Block, m=16 inducing points) |
| Bottleneck | global mean pool | **PMA** (Pooling by Multihead Attention, k=1 query) + global concat |
| Target head | per-template MLP | PMA-style cross-attention with **NUM_TEMPLATES learnable template queries** |
| 計算量 | O(P²) | O(P·m) |
| 出力形状 | (B, P, NUM_TEMPLATES) | (B, P, NUM_TEMPLATES) (case6 互換) |

## ディレクトリ構成

```
pipeline/imitation/case7/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # obs → torch.Tensor (case5 と同一、17 dim)
│   ├── timeline.py          # ship-prediction (case5 と同一)
│   ├── model.py             # ★ Set Transformer (DeepSetsPolicy alias)
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── templates.py         # target template 定義
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   ├── weights.pt           # canonical (DVC 管理、git untracked)
│   └── weights_smoke.pt     # 1-epoch smoke test 用 (DVC 管理)
├── configs/
│   ├── il_case7.yaml        # 本番設定
│   └── il_case7_smoke.yaml  # 1-epoch smoke 設定
├── training/                # 開発用 (.submitignore)
└── evaluation/              # 開発用 (.submitignore)
```

## 手順

```bash
cd bot

# 1) 前処理
uv run python -m pipeline.imitation.case7.training.preprocess \
    --config pipeline/imitation/case7/configs/il_case7.yaml

# 2) 学習
uv run python -m pipeline.imitation.case7.training.train \
    --config pipeline/imitation/case7/configs/il_case7.yaml

# 3) ローカル評価 (vs baseline_v1)
uv run python -m pipeline.imitation.case7.evaluation.eval_vs_baseline --episodes 50 --seed 0
```

## 設計原則

- **case 間独立**: case5/case6 と featurizer 完全同一だが import せず**コピー**保持。
- **Action 表現**: `(from_planet 多選択, target_template 分類, ships_bucket 分類)`
  の 3 ヘッド。`angle` は `aim_with_prediction()` で再構成。
- **モデル alias**: `model.py` の Set Transformer クラスは `DeepSetsPolicy` として
  alias 化されており、既存の agent.py / weight loader はそのまま動作する。

## レジストリ

```python
"il_v7": "pipeline.imitation.case7.policy.agent:agent",
```

## モデルバージョン

| ファイル | 説明 |
|---------|------|
| `policy/weights.pt` | canonical。`dev/{vast,runpod} promote` で上書き |
| `policy/weights_smoke.pt` | 1-epoch smoke test 用 (RunPod 動作確認向け) |
| `policy/weights_iter<N>.pt` | iteration 履歴 (生成された場合のみ。命名規則は imitation/README.md 参照) |
