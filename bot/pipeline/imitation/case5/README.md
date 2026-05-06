# Imitation Case5 — Ship-Prediction Featurizer

case1 (DeepSets baseline) の **planet 11 dim** に、`pipeline/rulebase/case6` から
portage した **ship-prediction timeline 6 列** を加えた imitation case。
バックボーンは Graph U-Net (kNN k=8, hidden=128, TopK pool 3 段)。

詳細: [`docs/experiment/imitation/20260503_case5_ship_prediction/`](../../../../docs/experiment/imitation/20260503_case5_ship_prediction/)

## case1 からの差分

| カテゴリ | case1 | case5 |
|----------|------:|------:|
| `PLANET_FEAT_DIM` | 11 | **17** (+6) |
| `GLOBAL_FEAT_DIM` | 6 | 6 (変更なし) |
| バックボーン | DeepSets | Graph U-Net |

### 追加された planet 特徴量 (列 11..16)

`policy/timeline.py` が各 planet について敵 fleet 到着順に owner / ships の推移を
シミュレートし、6 集約値を返す。オリジナルは
`bot/pipeline/rulebase/case6/baseline/core/world_model.py`
(`simulate_planet_timeline` / `normalize_arrivals` / `resolve_arrival_event`)。

| idx | 名前 | 概要 |
|-----|------|------|
| 11 | `loss_3turn` | 3 ターン以内の予測 ship loss |
| 12 | `ttf` | time-to-fall (敵手に渡るまでの予測ターン数) |
| 13 | `min_owned` | timeline 中の自軍最小所有 ships |
| 14 | `surplus` | 防衛に必要な ships を超過する余剰量 |
| 15 | `fall_predicted` | 期間内に敵手に渡る予測フラグ |
| 16 | `keep_needed` | 維持に必要な最小 ships |

## ディレクトリ構成

```
pipeline/imitation/case5/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # obs → torch.Tensor (17 dim)
│   ├── timeline.py          # ship-prediction (rulebase/case6 portage)
│   ├── model.py             # Graph U-Net policy (DeepSetsPolicy alias)
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── safety.py            # mission filter / no-op guard
│   ├── templates.py         # target template definitions
│   └── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
├── configs/
│   └── il_case5.yaml        # preprocess / training / evaluation 設定
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py        # replay → parquet
│   ├── dataset.py           # torch Dataset
│   ├── train.py             # BC 学習ループ
│   ├── losses.py            # 3-head focal/CE loss
│   └── expand_minority.py   # target/ships 少数派サンプル増強
└── evaluation/              # 開発用 (.submitignore)
    ├── eval_vs_baseline.py
    ├── replay_match.py
    ├── diagnose_weights.py
    ├── calibrate_heads.py
    ├── tune_thresholds.py
    └── plot_curves.py
```

## 手順

```bash
cd bot

# 1) データ前処理 (replay → parquet)
uv run python -m pipeline.imitation.case5.training.preprocess \
    --config pipeline/imitation/case5/configs/il_case5.yaml

# 2) 学習 (BC)
uv run python -m pipeline.imitation.case5.training.train \
    --config pipeline/imitation/case5/configs/il_case5.yaml

# 3) ローカル評価 (vs rulebase/case1 baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case5.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

## 設計原則

- **case 間独立**: `pipeline/imitation/case5/` は他 case に依存しない / されない。
  ship-prediction logic は rulebase/case6 から **コピー** (import ではない)。
- **Action 表現**: `(from_planet 多選択, target_template 分類, ships_bucket 分類)`
  の 3 ヘッド (case1/2/3 と同一)。`angle` は `aim_with_prediction()` で再構成。
- **推論 knobs**: `from_threshold=0.31`, `target_T=0.8`, `ships_T=1.1` (case1 と同値)。

## レジストリ

```python
"il_v5": "pipeline.imitation.case5.policy.agent:agent",
```

## モデルバージョン

| ファイル | 説明 |
|---------|------|
| `policy/weights.pt` | canonical。`agent.py` (`il_v5`) が読み込む。`dev/{vast,runpod} promote` で上書き |
| `policy/weights_iter<N>.pt` | iteration 履歴 (生成された場合のみ) |
