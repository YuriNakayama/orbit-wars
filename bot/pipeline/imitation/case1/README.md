# Imitation Case1 — Imitation Learning Baseline

過去の Kaggle 上位リプレイ (`data/lake/kaggle_episodes/matches/`) からの **行動クローニング (BC)** で
動く PyTorch エージェント。`pipeline/rulebase/case1` (rule-based) との 1v1 勝率 ≥ 50% を目標とする。

設計詳細は [`docs/plans/imitation-learning-baseline/`](../../../docs/plans/imitation-learning-baseline/) を参照。

## ディレクトリ構成

```
pipeline/imitation/case1/
├── main.py                  # Kaggle entry (Path.cwd() ベース)
├── policy/                  # 提出物
│   ├── agent.py             # agent(obs) エントリ
│   ├── featurizer.py        # obs → torch.Tensor
│   ├── model.py             # DeepSets policy
│   ├── decoder.py           # PolicyOutput → action list
│   ├── geometry.py          # aim_with_prediction (独立コピー)
│   └── weights.pt           # 学習済み重み (DVC 管理、git untracked)
├── training/                # 開発用 (.submitignore)
│   ├── preprocess.py        # replay → parquet
│   ├── dataset.py           # torch Dataset
│   ├── train.py             # BC 学習ループ
│   └── losses.py            # 3-head CE loss
└── evaluation/              # 開発用 (.submitignore)
    └── eval_vs_baseline.py
```

ハイパーパラメータはリポジトリルートの `params.yaml` に集約されており、
CLI は `--config` 引数を持たず常に `params.yaml` を読みます。

## 手順

### DVC 経由 (推奨)

```bash
# 全 pipeline を依存グラフで再実行（data/weights が最新なら skip）
uv run --directory backend dvc repro

# 単一 stage のみ
uv run --directory backend dvc repro preprocess_imitation_case1
uv run --directory backend dvc repro train_imitation_case1
uv run --directory backend dvc repro eval_imitation_case1
```

### 直接実行

```bash
cd backend

# 1) データ前処理 (replay → parquet)
uv run python -m pipeline.imitation.case1.training.preprocess

# 2) 学習 (BC) — weights.pt を上書き
uv run python -m pipeline.imitation.case1.training.train

# 3) ローカル評価 (vs rulebase/case1 baseline_v1, 100 戦)
uv run python -m pipeline.imitation.case1.evaluation.eval_vs_baseline --episodes 100 --seed 0
```

パラメータを変更したい場合はリポジトリルートの `params.yaml` を編集します。

### Vast.ai 経由 (GPU 学習、ローカルから直接起動)

```bash
git push origin <branch>
dev/vast train <commit-sha>            # search offers → pick → onstart 起動
dev/vast pull <run_id>                 # 完了後にローカル取得
dev/vast promote <run_id>              # 採用なら canonical weights に昇格
```

`train.py` は `ORBIT_WARS_RUN_DIR` env が指定されているとき、その下に `best.pt`、
`metrics.json`、`run.json` を出力する設計です（canonical `policy/weights.pt` は
`dev/vast promote` を経由しないと更新されません）。詳細は
[`docs/plans/vast-ai-basis/`](../../../../docs/plans/vast-ai-basis/) を参照。

## テスト

```bash
uv run pytest tests/pipeline/imitation/case1 -v -m "not slow"
uv run pytest tests/pipeline/imitation/case1 -v   # determinism 含む
```

## 設計原則

- **case 間独立**: `pipeline/imitation/case1/` は `pipeline/rulebase/case[012]/` に依存しない / されない。
- **Action 表現**: `(from_planet 分類, target_planet 分類, ships_bucket 分類)` の 3 ヘッド。
  `angle` は `aim_with_prediction()` で決定論的に再構成。
- **推論**: greedy argmax + 有効ターゲットマスク + from_threshold で no-op 判定。
- **モデル**: DeepSets (惑星集合の順不同 invariant), hidden=64, weights < 1MB。

## 評価結果 (2026-04-19)

旧 baseline (10 epoch) は `il_v1` vs `baseline_v1` で win_rate 0.00。
診断書 [`docs/experiment/imitation/20260419_case1_diagnosis/result.md`](../../../docs/experiment/imitation/20260419_case1_diagnosis/result.md)
の Bug 1/3/4 + 追加バグを以下のとおり修正済み:

- preprocess: 1 フレーム = 1 row、`from_multihot` + `target_per_src` + `ships_per_src`
  形式。target は `aim_with_prediction` 順方向逆解決。敗者側 obs も採用。
- preprocess (新規): kaggle replay の **loser 側 obs.step / obs.player が None**
  になっているため、`step_idx` と `slot` から注入。これを直さないと敗者フレームが
  全て step=0 として扱われ、序盤特徴量が壊滅的に汚染される。
- losses: from-head は my_planets に限定した multi-hot BCE。target/ships は
  発射 source の per-row CE。
- model: target_pair に (dx, dy, distance, ship_log_diff, tgt_is_enemy,
  tgt_is_neutral) の pairwise 幾何特徴を追加し、対角 (src=tgt) を -inf マスク。
- decoder: ships バケットを 4 分類 (25/50/75/100%) に変更し、`bucket=0` でも
  最低 1 艦は送る。
- config: `from_threshold=0.05` (n_my=1 の serve prior 0.14 を考慮し低く),
  `target_w=2.0`, `rating_quantile=0.5`, `epochs=15`, `modes=["1v1"]`。

vs random / vs noop は安定して勝てるが、vs `rulebase/case1` baseline_v1 は
依然 0/100 (Apr 19)。BC 単体ではタクティカル決定力に課題が残るため、
target/ships head の精度向上 (val_target_acc=0.34) が次の改善ポイント。
