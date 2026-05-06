# Imitation Case2 — Feature-Engineered IL

`pipeline/imitation/case1` をベースに、観測から抽出する planet/global 特徴量を
強化した模倣学習エージェント。アーキテクチャ (Graph U-Net + 3 head) はそのまま、
入力次元のみ拡張する純粋な特徴量エンジニアリング派生。

## case1 からの差分

| カテゴリ | case1 | case2 |
|----------|------:|------:|
| `PLANET_FEAT_DIM` | 11 | **18** (+7) |
| `GLOBAL_FEAT_DIM` | 6 | **11** (+5) |
| `TEMPLATE_CTX_DIM` | 40 | 40 (変更なし) |

### 追加された planet 特徴量 (列 11..17)

| idx | 名前 | 概要 |
|-----|------|------|
| 11 | `sun_dist_norm` | 太陽中心からの距離 / sqrt(2)·BOARD_SIZE |
| 12 | `is_static` | 静止軌道惑星フラグ (rotation lock) |
| 13 | `prod_per_ship` | production / max(1, ships) (clip [0,5]/5) |
| 14 | `nearest_enemy_dist` | 最寄り敵惑星までの距離 / DIAG |
| 15 | `support_density` | 25u 以内の自軍 ships 合計 (log1p / 6) |
| 16 | `threat_density` | 25u 以内の敵 ships+fleets 合計 (log1p / 6) |
| 17 | `net_incoming_signed` | (敵-味方 incoming) / max(1, ships) (clip ±3) |

### 追加された global 特徴量 (列 6..10)

| idx | 名前 | 概要 |
|-----|------|------|
| 6 | `my_planet_frac` | 自軍惑星数 / 全惑星数 |
| 7 | `enemy_planet_frac` | 敵惑星数 / 全惑星数 |
| 8 | `comet_active` | step が彗星ウェーブ窓 (50/150/250/350/450 + 30) 内か |
| 9 | `phase_mid` | 100 ≤ step < 300 |
| 10 | `phase_late` | step ≥ 300 |

これらは case1 の評価診断 (`docs/experiment/imitation/20260419_case1_diagnosis/result.md`)
で「target/ships head の精度が頭打ち」だった原因として、序盤・中盤・終盤の
形勢、空間的脅威、生産密度といった**構造化された戦略文脈**が不足している点を
補うためのもの。

## ディレクトリ構成

```
pipeline/imitation/case2/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # 拡張 obs → torch.Tensor
│   ├── model.py             # Graph U-Net policy (case1と同形)
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── templates.py         # target template definitions
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   ├── weights.pt           # canonical (= weights_iter1.pt のコピー)
│   ├── weights_iter0.pt     # baseline iter (旧 weights_baseline.pt)
│   └── weights_iter1.pt     # phase1 iter (旧 weights_phase1.pt)
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py        # replay → parquet
│   ├── dataset.py           # torch Dataset
│   ├── train.py             # BC 学習ループ
│   └── losses.py            # 3-head focal/CE loss
├── evaluation/              # 開発用 (.submitignore)
│   └── eval_vs_baseline.py
└── configs/                 # 開発用 (.submitignore)
    └── il_baseline.yaml
```

## 手順

```bash
# 1) データ前処理 (replay → parquet)
uv run python -m pipeline.imitation.case2.training.preprocess \
    --config pipeline/imitation/case2/configs/il_baseline.yaml

# 2) 学習 (BC)
uv run python -m pipeline.imitation.case2.training.train \
    --config pipeline/imitation/case2/configs/il_baseline.yaml

# 3) ローカル評価 (vs rulebase/case1 baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case2.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

## テスト

```bash
uv run pytest tests/pipeline/imitation/case2 -v -m "not slow"
uv run pytest tests/pipeline/imitation/case2 -v   # determinism / integration 含む
```

## 設計原則

- **case 間独立**: `pipeline/imitation/case2/` は `pipeline/imitation/case1/` および
  `pipeline/rulebase/case[012]/` に依存しない / されない。
  全コードは case2 配下で完結する独立コピー。
- **アーキテクチャ非変更**: Graph U-Net (kNN k=8, hidden=128, TopK pool 2/3 → 1/2)
  と 3-head 出力構造は case1 と同一。weight 構造は `planet_in_dim` / `global_in_dim`
  のみが異なる。
- **Action 表現**: `(from_planet 分類, target_template 分類, ships_bucket 分類)`
  の 3 ヘッド。`angle` は `aim_with_prediction()` で決定論的に再構成。
- **推論**: greedy argmax + 有効ターゲットマスク + from_threshold で no-op 判定
  (case1 と同じ knobs: `from_threshold=0.31`, `target_T=0.8`, `ships_T=1.1`)。

## レジストリ

`src/dataset/selfplay/agents.py` に `il_v2` / `il_v2_phase1` として登録:

```python
"il_v2":         "pipeline.imitation.case2.policy.agent:agent",
"il_v2_phase1":  "pipeline.imitation.case2.policy.agent_phase1:agent",
```

## モデルバージョン

| ファイル | 説明 |
|---------|------|
| `policy/weights.pt` | canonical。`agent.py` (`il_v2`) が読み込む |
| `policy/weights_iter0.pt` | baseline iter (旧 `weights_baseline.pt`) |
| `policy/weights_iter1.pt` | phase1 iter (旧 `weights_phase1.pt`)。`agent_phase1.py` (`il_v2_phase1`) が読み込む |

ablation variant (`weights_phase1_no_a..no_e`, `weights_phase1_clean_c`) は `configs/il_phase1_*.yaml` 経由で出力される一時アーティファクト。`policy/` 配下には保持しない。
