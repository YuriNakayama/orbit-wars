# Imitation Case8 — Feature Engineering (57 × 8)

case7 系列 (Set Transformer 互換 backbone) を base に、**feature engineering を大幅拡張**した
imitation case。iter4 で planet 61 / global 12 まで膨らませ、iter6 の K2/K4 ablation で
causal leak / dead group を削除し **planet 57 / global 8** に確定した版。

詳細: [`docs/experiment/imitation/20260504_case8_feature_engineering/`](../../../../docs/experiment/imitation/20260504_case8_feature_engineering/)

## 特徴量次元の変遷

| Iter | PLANET_FEAT_DIM | GLOBAL_FEAT_DIM | 概要 |
|------|----------------:|----------------:|------|
| iter4 | 61 | 12 | feature 集合フル拡張 |
| iter5 | 61 | 12 | permutation importance で K2/K4 を検出 |
| iter6 (現状) | **57** | **8** | K2 削除 (4 列) + K4 削除 (4 列) |

### iter6 で削除した dim group

| Group | 削除元 | 削除理由 |
|-------|--------|---------|
| K2 outgoing_fleet_trajectory (planet 53–56, 4 列) | planet | **Causal leak**: `obs.fleets` は action_N 適用後の値。自軍 fleet の `from_pid` / `ships` が `ships_per_src` ラベルと完全一致し、policy が逆算可能になる。iter5 permutation で `sum|Δ|=1.70` |
| K4 aux_multi_horizon_global (global 8–11, 4 列) | global | iter5 permutation で `sum|Δ|=0.001`、H2 と相関で意義なし dead group |

K3 frontline_distance は slot shift で planet 57→53 へ移動。

## case7 からの差分

| カテゴリ | case7 | case8 |
|----------|------:|------:|
| `PLANET_FEAT_DIM` | 17 | **57** |
| `GLOBAL_FEAT_DIM` | 6 | **8** |
| バックボーン | Set Transformer | Graph U-Net (DeepSetsPolicy alias) |
| history 取扱い | obs_N | **obs_{N-2}, obs_{N-3}** のみ (causal leak 対策) |
| launch event | obs.fleets 直参照 | `prev_fleet_snapshot` 差分 (`prev_fleets_{N-2..N-5}` の N−1 と N−2 比較) |

## ディレクトリ構成

```
pipeline/imitation/case8/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ + HistoryState ring buffer
│   ├── featurizer.py        # obs → torch.Tensor (57 × 8)
│   ├── timeline.py          # ship-prediction + multi-horizon summary
│   ├── model.py             # Graph U-Net policy (DeepSetsPolicy alias)
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── templates.py         # target template 定義
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   └── weights.pt           # canonical (DVC 管理、git untracked)
├── configs/
│   └── il_case8.yaml        # 本番設定
├── training/                # 開発用 (.submitignore)
└── evaluation/              # 開発用 (.submitignore)
```

## 手順

```bash
cd bot

# 1) 前処理 (case8 専用 parquet を data/mart/imitation/case8/ に生成)
uv run python -m pipeline.imitation.case8.training.preprocess \
    --config pipeline/imitation/case8/configs/il_case8.yaml

# 2) 学習 (BC) — weights.pt を上書き
uv run python -m pipeline.imitation.case8.training.train \
    --config pipeline/imitation/case8/configs/il_case8.yaml

# 3) ローカル評価 (vs rulebase/case1 baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case8.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

## DVC 経由

```bash
uv run --directory bot dvc repro preprocess_imitation_case8
uv run --directory bot dvc repro train_imitation_case8
uv run --directory bot dvc repro eval_imitation_case8
```

## 設計原則

- **case 間独立**: case7 と featurizer 構造は近いが、history / leak 対策の差が大きいため独立コピー。
- **Causal leak 対策**: action_N 適用後の `obs.fleets` を直接見ず、obs_{N-2}/obs_{N-3} と
  `prev_fleet_snapshot` 差分のみを参照する。HistoryState は per-match ring buffer。
- **Action 表現**: `(from_planet 多選択, target_template 分類, ships_bucket 分類)` の 3 ヘッド。

## レジストリ

```python
"il_v8": "pipeline.imitation.case8.policy.agent:agent",
```

## モデルバージョン

| ファイル | 説明 |
|---------|------|
| `policy/weights.pt` | canonical。`dev/{vast,runpod} promote` で上書き |
| `policy/weights_iter<N>.pt` | iteration 履歴 (生成された場合のみ。命名規則は imitation/README.md 参照) |
