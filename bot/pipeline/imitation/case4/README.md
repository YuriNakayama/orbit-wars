# Imitation Case4 — Kaggle tutorial-style candidate head (BC)

`pipeline/imitation/case3` の Graph U-Net バックボーンと history feature 群を
そのまま受け継ぎ、出力 head を Kaggle ノートブック
[`kashiwaba/orbit-wars-reinforcement-learning-tutorial`](https://www.kaggle.com/code/kashiwaba/orbit-wars-reinforcement-learning-tutorial)
に倣った **per-source × candidate categorical** に差し替えた模倣学習エージェント。

## case3 からの差分

| カテゴリ | case3 | case4 |
|---|---|---|
| Backbone | Graph U-Net (kNN k=8, hidden=128, TopK pool 3 段) | **完全同一** |
| `PLANET_FEAT_DIM` | 35 | 35 (変更なし) |
| `GLOBAL_FEAT_DIM` | 20 | 20 (変更なし) |
| Per-source context | `template_ctx` (40 dim) | **`candidate_feats` (K=8 × 14 dim)** |
| `from_head` | per-planet sigmoid (P,) | **削除** (slot 0 で no-op 表現) |
| `target_head` | per-planet × NUM_TEMPLATES=8 | **削除** |
| `ships_head` | per-planet × 4 buckets | **削除** (rule-based) |
| `candidate_head` | — | **新設**: per-source K=8 categorical |
| Ships rule | learned (4 buckets) | `max(target.ships+1, 20)` (notebook 流) |

## Action / observation 設計

ノートブックの per-source × K candidate 構造を BC で再現:

- `target_index = 0` → no-op (この惑星は撃たない)
- `target_index = 1..K-1` → 候補スロット (距離順 enemy / neutral / friendly bucket)

候補ビルダーは `policy/candidates.py` 参照: enemy_quota = neutral_quota = ⌊(K-1)/3⌋,
残りは friendly。足りなければ近距離順 fallback。

per-candidate 14 次元 (notebook と同形):

| idx | 名前 | 概要 |
|---|---|---|
| 0 | `is_valid` | スロットが populated か |
| 1..3 | `is_neutral / is_mine / is_enemy` | 候補の所有者 1-hot |
| 4..5 | `tgt.x, tgt.y` | (board_size 正規化) |
| 6..7 | `dx, dy` | src→target 相対座標 |
| 8 | `dist` | src→target 距離 (board_size 正規化) |
| 9 | `tgt.ships` | (max_ships=400 正規化) |
| 10 | `tgt.production` | (max=5 正規化) |
| 11 | `tgt_is_rotating` | 軌道惑星か |
| 12 | **`crosses_sun`** | src→target 直線が sun に被るか (case3 にない新シグナル) |
| 13 | `src.ships` | reference signal |

slot 0 (no-op) は `is_valid=1.0` 以外 0.0。

`candidate_mask`: slot 0 = 常に True。slot 1..K-1 は
`ships_needed > 0 AND not crosses_sun AND src.ships >= ships_needed` で True。
無効 slot は logit を `-inf` でマスクする。

## ディレクトリ構成

```
pipeline/imitation/case4/
├── main.py                  # Kaggle entry (sys.path.insert(0, str(Path.cwd())))
├── policy/
│   ├── agent.py             # agent(obs) — HistoryState を per-match 保持
│   ├── candidates.py        # build_candidates / candidate_features (notebook 流)
│   ├── featurizer.py        # planet 35 / global 20 / candidate 8x14
│   ├── geometry.py          # aim_with_prediction (case3 から独立コピー)
│   ├── decoder.py           # candidate_logits → action list (rule-based ships)
│   ├── model.py             # Graph U-Net + per-source candidate head
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   └── weights.pt           # 学習済み重み (training 後生成)
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py
│   ├── dataset.py
│   ├── losses.py
│   └── train.py
├── evaluation/              # 開発用 (.submitignore)
│   └── eval_vs_baseline.py
└── configs/                 # 開発用 (.submitignore)
    └── il_case4.yaml
```

## 手順

```bash
# 1) データ前処理 (replay → parquet)
uv run --directory backend python -m pipeline.imitation.case4.training.preprocess \
    --config pipeline/imitation/case4/configs/il_case4.yaml

# 2) 学習 (BC, ローカル)
uv run --directory backend python -m pipeline.imitation.case4.training.train \
    --config pipeline/imitation/case4/configs/il_case4.yaml

# 2') GPU 学習 (Vast.ai)
git push origin <branch>
dev/vast train <commit-sha> --case case4 --stage train_imitation_case4

# 3) ローカル評価 (vs rulebase/case1 baseline_v1)
uv run --directory backend python -m pipeline.imitation.case4.evaluation.eval_vs_baseline \
    --episodes 30 --seed 0
```

## レジストリ

`src/dataset/selfplay/agents.py` に `il_v4` として登録:

```python
"il_v4": "pipeline.imitation.case4.policy.agent:agent",
```

## 設計原則

- **case 間独立**: case4 は case1/case2/case3 / rulebase/* に依存しない独立コピー。
  shared なヘルパー (`geometry.py` 等) は **直接コピー** している。
- **Backbone 不変**: Graph U-Net (kNN k=8, hidden=128, TopK pool 3 段) は case3 と
  完全一致させる (user 指示)。head 部のみ再設計。
- **Action 表現**: `(per-source candidate slot)` の 1 head。slot 0 は no-op、
  slot 1..7 は notebook 流の候補リストへ写像。`ships` は `max(tgt.ships+1, 20)` 固定。
- **Evaluation**: Kaggle publicScore は使わない。ローカル `eval_vs_baseline` の勝率のみ。
  300 戦未満の結果は noise として扱う (`project_imitation_case1_phase3` の知見)。
