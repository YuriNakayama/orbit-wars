# Case3 — Imitation Learning Baseline

過去の Kaggle 上位リプレイ (`data/kaggle_episodes/matches/`) からの **行動クローニング (BC)** で
動く PyTorch エージェント。`pipeline/rulebase/case1` (rule-based) との 1v1 勝率 ≥ 50% を目標とする。

設計詳細は [`docs/plans/imitation-learning-baseline/`](../../../docs/plans/imitation-learning-baseline/) を参照。

## ディレクトリ構成

```
pipeline/imitation/case3/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # obs → torch.Tensor
│   ├── model.py             # DeepSets policy
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (case3 独立コピー)
│   └── weights.pt           # 学習済み重み
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py        # replay → parquet
│   ├── dataset.py           # torch Dataset
│   ├── train.py             # BC 学習ループ
│   └── losses.py            # 3-head CE loss
├── evaluation/              # 開発用 (.submitignore)
│   └── eval_vs_baseline.py
└── configs/                 # 開発用 (.submitignore)
    └── il_baseline.yaml
```

## 手順

```bash
# 1) データ前処理 (replay → parquet)
uv run python -m pipeline.imitation.case3.training.preprocess --config pipeline/imitation/case3/configs/il_baseline.yaml

# 2) 学習 (BC)
uv run python -m pipeline.imitation.case3.training.train --config pipeline/imitation/case3/configs/il_baseline.yaml

# 3) ローカル評価 (vs case1_baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case3.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

## テスト

```bash
uv run pytest tests/pipeline/imitation/case3 -v -m "not slow"
uv run pytest tests/pipeline/imitation/case3 -v   # determinism 含む
```

## 設計原則

- **case 間独立**: `pipeline/imitation/case3/` は `pipeline/rulebase/case[012]/` に依存しない / されない。
- **Action 表現**: `(from_planet 分類, target_planet 分類, ships_bucket 分類)` の 3 ヘッド。
  `angle` は `aim_with_prediction()` で決定論的に再構成。
- **推論**: greedy argmax + 有効ターゲットマスク + from_threshold で no-op 判定。
- **モデル**: DeepSets (惑星集合の順不同 invariant), hidden=64, weights < 1MB。

## 評価結果 (2026-04-19)

| 対戦相手 | episodes | wins | losses | draws | win_rate |
|---|---|---|---|---|---|
| `baseline_v1` (1v1) | 100 | 0 | 100 | 0 | **0.00** |
| `random` (1v1, smoke) | 20 | 2 | 18 | 0 | 0.10 |

`weights.pt` (10 epoch 学習) は val_loss 2.56 (from_acc 0.49 / target_acc 0.37 / ships_acc 0.72)。
target head の精度が不十分で、推論時にほぼ no-op が選ばれる。発射閾値を `from_threshold=0.05`
まで下げても baseline には全敗で **目標 ≥50% 未達**。

### 想定される改善方向

- データ拡張: 現状 100k frame (top 25% rating)。閾値緩和で 200-300k frame に増やす。
- 損失バランス: `target_w` を 2.0 程度まで上げて argmax 多様性を確保。
- アーキテクチャ: `target_pair` head に attention を導入し pairwise 表現を強化。
- カリキュラム: BC 後に self-play fine-tune (RL) で baseline の戦術に適応。
- ラベル品質: `_fleet_target_planet_id()` の解決失敗 (no-op 25%) を再検討、明示的な no-op
  サンプリングを減らす。
