# Imitation Case3 — Phase 2 (history features + Graph U-Net)

`pipeline/imitation/case2` をベースに、観測履歴から取り出す**短期時系列特徴量**を
追加し、バックボーンを DeepSets から **Graph U-Net** に置き換えた模倣学習エージェント。
case2 の特徴量 (planet 18 / global 11) からさらに次元を拡張した派生で、case 間独立原則に従い
case3 配下で完結する独立コピー。

## case2 からの差分

| カテゴリ | case2 | case3 |
|----------|------:|------:|
| `PLANET_FEAT_DIM` | 18 | **35** (+17) |
| `GLOBAL_FEAT_DIM` | 11 | **20** (+9) |
| `TEMPLATE_CTX_DIM` | 40 | 40 (変更なし) |
| Backbone | DeepSets | **Graph U-Net** (kNN k=8, TopK pool 3 段) |

case3 の planet/global 拡張は 2 段階で構成される。

### Phase 1 拡張 (case2 baseline 18 → 32, group D は採用見送り)

`featurizer_phase2.py` の docstring 通り、cols 18..31 は phase1 で追加した
構造化文脈特徴量。group A/B/C のみ採用、threat_pressure_long (group D) は
**実験で寄与なし**だったため除外。

| idx | 名前 | グループ | 概要 |
|-----|------|----------|------|
| 18 | `nearest_ally_dist` | A | 最寄り自軍惑星までの距離 / DIAG |
| 19 | `nearest_neutral_dist` | A | 最寄り中立惑星までの距離 / DIAG |
| 20..27 | `orbit_dx/dy @ t={1,2,4,8}` | B | 軌道予測の相対位置 (4 horizon × dx,dy) |
| 28 | `incoming_ally_eta` | C | 自軍 fleet の最短 ETA / HORIZON |
| 29 | `incoming_enemy_eta` | C | 敵 fleet の最短 ETA / HORIZON |
| 30 | `incoming_ally_ships` | C | 自軍 fleet の合計 ships (log1p / 6) |
| 31 | `incoming_enemy_ships` | C | 敵 fleet の合計 ships (log1p / 6) |

### Phase 2 拡張 (32 → 35, planet 履歴)

`HistoryState.prev_planet_snapshots` (deque maxlen=3) を読み、各惑星の
過去状態との差分を取り込む。

| idx | 名前 | 概要 |
|-----|------|------|
| 32 | `delta_ships_t1` | (ships_now − ships_{t-2}) / max(1, ships_now)、clip ±1 |
| 33 | `delta_ships_t2` | (ships_now − ships_{t-3}) / max(1, ships_now)、clip ±1 |
| 34 | `owner_changed_t1` | 2 ターン前と所有者が違うか (0/1) |

> **ラベルリーク回避**: `obs[N].observation` は **step N の post-action 状態** なので、
> `prev[-1]` (= 1 ターン前 obs) との差分は action_N の発射 ships を直接含む。
> よって履歴は `[-2]` / `[-3]` を読んで 2 ターン以上ずらす。詳細は
> `featurizer_phase2.py` の `HistoryState` docstring を参照。

### Phase 1 拡張 (global 11 → 16, group D 落ち合わせ)

cols 0..10 は case2 baseline と同じ 11 dim、11..15 が phase1 で追加した 5 dim。

| idx | 名前 | 概要 |
|-----|------|------|
| 11 | `next_comet_eta_norm` | 次の彗星ウェーブ到達までのターン数 / 100 (clip 1.0) |
| 12 | `my_ships_share` | 自軍 ships / 全 ships |
| 13 | `enemy_ships_share` | 敵 ships / 全 ships |
| 14 | `my_prod_share` | 自軍 production / 全 production |
| 15 | `score_diff_log` | (log1p(my_ships) − log1p(enemy_ships)) / 3 (clip ±1) |

### Phase 2 拡張 (16 → 20, 直近 4 ターンの launch 履歴)

`HistoryState.recent_launches` (deque maxlen=200) を集計。

| idx | 名前 | 概要 |
|-----|------|------|
| 16 | `enemy_launch_count_last4` | 敵が直近 4 ターンに発射した艦隊数 / 10 |
| 17 | `enemy_launch_ships_last4` | 敵が直近 4 ターンに発射した ships 合計 (log1p / 6) |
| 18 | `ally_launch_count_last4` | 自軍が直近 4 ターンに発射した艦隊数 / 10 |
| 19 | `ally_launch_ships_last4` | 自軍が直近 4 ターンに発射した ships 合計 (log1p / 6) |

## モデル: Graph U-Net (`policy/model.py`)

case1/case2 の DeepSets を捨てて Graph U-Net (Gao & Ji 2019) に変更:

- 各フレームの planet 座標 (x, y) から kNN グラフ構築 (`KNN_K = 8`, padding 行は孤立)
- Encoder 3 段: `GraphConv → TopK pool` (`P → P*2/3 → P/3`)
- Bottleneck: GraphConv + global mean pool → context vector
- Decoder 2 段: unpool (scatter back) + skip-add + GraphConv
- 出力 3 head は case1/case2 と同形:
  - `from_planet` 分類 (sigmoid, P 二値)
  - `target_template` 分類 (P × NUM_TEMPLATES)
  - `ships_bucket` 分類 (P × ships_buckets=4)

クラス名は **既存 weights/import 互換のため `DeepSetsPolicy` のまま alias**
として残してある (中身は完全に Graph U-Net)。

## ディレクトリ構成

```
pipeline/imitation/case3/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent_phase2.py      # agent(obs) — HistoryState を per-match 保持
│   ├── featurizer_phase2.py # 拡張 obs + 履歴 → torch.Tensor (563 行)
│   ├── model.py             # Graph U-Net policy
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   ├── templates.py         # target template definitions
│   ├── types.py             # BatchFeatures / PolicyOutput / WorldSnapshot
│   └── weights_phase2.pt    # 学習済み重み (training 後生成)
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py        # replay → parquet (history 同梱)
│   ├── dataset.py           # torch Dataset
│   ├── train.py             # BC 学習ループ (Vast.ai 対応, run_dir 自動採番)
│   ├── losses.py            # 3-head focal/CE loss + class weights
│   └── expand_minority.py   # target/ships の少数派サンプル増強
├── evaluation/              # 開発用 (.submitignore)
│   ├── eval_metrics.py
│   └── eval_vs_baseline.py
├── scripts/                 # 開発用 (.submitignore)
│   ├── calibrate_heads.py
│   └── tune_thresholds.py
└── configs/                 # 開発用 (.submitignore)
    └── il_phase2.yaml
```

## 手順

```bash
# 1) データ前処理 (replay → parquet, history 同梱)
uv run python -m pipeline.imitation.case3.training.preprocess \
    --config pipeline/imitation/case3/configs/il_phase2.yaml

# 2) 学習 (BC)
uv run python -m pipeline.imitation.case3.training.train \
    --config pipeline/imitation/case3/configs/il_phase2.yaml

# 2') GPU 学習 (Vast.ai)
git push origin <branch>
dev/vast train <commit-sha> --stage train_imitation_case3

# 3) ローカル評価 (vs rulebase/case1 baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case3.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

## テスト

```bash
uv run pytest tests/pipeline/imitation/case3 -v -m "not slow"
uv run pytest tests/pipeline/imitation/case3 -v   # determinism / integration 含む
```

## 設計原則

- **case 間独立**: `pipeline/imitation/case3/` は `pipeline/imitation/case[12]/` および
  `pipeline/rulebase/**` に依存しない / されない。全コードは case3 配下で完結する独立コピー。
- **Backbone 変更**: case1/case2 の DeepSets → **Graph U-Net** (kNN k=8, hidden=128,
  TopK pool 3 段)。出力 3 head の形は case1/case2 と同一なので decoder/templates は流用できる。
- **History 管理**: preprocess (per-episode) と推論 agent (per-match) で同じ
  `HistoryState` 構造体を使い回す。agent 側は `obs.step == 0` または step 退行を
  検知して自動 reset。
- **Action 表現**: `(from_planet 分類, target_template 分類, ships_bucket 分類)`
  の 3 ヘッド。`angle` は `aim_with_prediction()` で決定論的に再構成。
- **推論 knobs** (`configs/il_phase2.yaml` `inference:`): `from_threshold=0.31`,
  `target_temperature=0.8`, `ships_temperature=1.1` (case1/case2 と同値)。
  環境変数 `IL_FROM_THRESHOLD` / `IL_MIN_FIRE_TOPK` / `IL_MAX_FIRE_COUNT` で上書き可能。

## レジストリ

`src/dataset/selfplay/agents.py` に `il_v3` として登録:

```python
"il_v3": "pipeline.imitation.case3.policy.agent_phase2:agent",
```

## 学習結果 (run `20260430-075004 / 0e9337d / seed0`)

Vast.ai (RTX 3090) で 15 epoch 学習した最新ラン。`best.pt` は **epoch 9, val_total=3.6911**。

| epoch | train_total | val_total | val_from F1\* | val_target top1 | val_target macro-F1 | val_ships top1 |
|------:|------------:|----------:|--------------:|----------------:|--------------------:|---------------:|
| 0     | 3.993 | 3.921 | 0.538 | 0.370 | 0.225 | 0.764 |
| 3     | 3.756 | 3.745 | 0.596 | 0.400 | 0.260 | 0.830 |
| 6     | 3.690 | 3.709 | 0.608 | 0.415 | 0.305 | 0.839 |
| **9** ★ | 3.640 | **3.691** | 0.615 | **0.419** | 0.306 | 0.837 |
| 12    | 3.602 | 3.698 | 0.615 | 0.415 | 0.310 | 0.837 |
| 14    | 3.576 | 3.715 | 0.613 | 0.420 | 0.314 | 0.833 |

- train loss は単調減少だが val loss は **epoch 9 で底打ち**、以降は微増 (= ごく軽い overfit)
- from head: ROC-AUC 0.93 / PR-AUC 0.63 / F1\* 0.61 で頭打ち
- target head: top1 ≈ 0.42、top2 ≈ 0.63、macro-F1 0.31 程度に収束。
  Phase 2 履歴を入れても target head は依然として頭打ちで、macro-F1 が低いことから
  少数派 template が依然として学習しきれていない (`expand_minority.py` は使ったがまだ不十分)
- ships head: top1 0.84 / macro-F1 0.67、ordinal MAE 0.25 で安定

学習履歴の生データは `data/output/models/imitation/case3/runs/<run_id>/history.jsonl`、
要約は `summary.json` / `run.json` に保存され、DVC + S3 で永続化されている。
