# Imitation Learning Baseline (case3) — Implementation Steps

実装順序は **データ前処理 → モデルと学習 → 推論と評価** の 3 フェーズ。1 タスク = 1 ファイル・1 コンポーネントの粒度で計 11 ステップに分解する。

## Phase 1: データ前処理 (Step 1-4)

### Step 1: プロジェクト雛形とコンフィグ

**Target**: pipeline / configs
**Dependencies**: なし

**概要**: `pipeline/case3/` の空ディレクトリ構造と `configs/il_baseline.yaml` を作成する。PyTorch 依存を `pyproject.toml` に追加する。

**Work Items**:
- [ ] `pipeline/case3/__init__.py`, `policy/__init__.py`, `training/__init__.py`, `evaluation/__init__.py` を作成。
- [ ] `pipeline/case3/configs/il_baseline.yaml` を作成 (model/train/data ハイパーパラメータ)。
- [ ] `pipeline/case3/README.md` 雛形 (目的・手順)。
- [ ] `pyproject.toml` に `torch>=2.3.0` を追加。
- [ ] `pipeline/.submitignore` に `training/` を追記 (`evaluation/`, `configs/` は既存)。
- [ ] `uv sync` で依存解決確認。

**Target Files (Expected)**:
- `pipeline/case3/__init__.py`, `pipeline/case3/configs/il_baseline.yaml`, `pipeline/case3/README.md`
- `pyproject.toml`, `pipeline/.submitignore`

**Acceptance Criteria**:
- `uv run python -c "import torch; print(torch.__version__)"` が成功。
- `uv run ruff check pipeline/case3` が 0 warning。
- `pipeline/case3/configs/il_baseline.yaml` が YAML として valid (`uv run python -c "import yaml; yaml.safe_load(open('...'))"`)。

---

### Step 2: データ前処理ロジック (preprocess.py)

**Target**: training
**Dependencies**: Step 1

**概要**: `data/kaggle_episodes/` のリプレイから rating_mu top25% + winner フィルタで学習用フレームを抽出し、`data/lake/case3/{train,val}.parquet` に書き出す。

**Work Items**:
- [ ] `pipeline/case3/training/preprocess.py` 実装: `preprocess(cfg) -> PreprocessReport`。
  - `data/kaggle_episodes/matches/index.parquet/mode=*/*.parquet` を polars で読み込み。
  - `agent_{0,1}_rating_mu` の quantile(0.75) を cutoff として、`winner` 側の (match_id, player, submission_id, rating_mu) を抽出。
  - episode 単位で 90/10 に train/val split (match_id ハッシュで deterministic)。
  - 各 replay を `gzip` 展開し `steps[*][player].observation/action` をフレーム展開。
  - obs → per-planet 特徴 (11ch) + global 特徴 (6ch) を numpy で構築。
  - action リスト展開で 1 フレーム 1 action = 1 行 (no-op も 1 行)。
- [ ] `training/cli.py` (または `__main__.py`) で Typer ベース CLI: `uv run python -m pipeline.case3.training.preprocess --config configs/il_baseline.yaml`。
- [ ] 前処理結果の summary 出力 (frames 数, train/val ratio, rating cutoff 値)。

**Target Files (Expected)**:
- `pipeline/case3/training/preprocess.py`
- `pipeline/case3/training/__main__.py` (CLI エントリ)
- `data/lake/case3/train.parquet`, `data/lake/case3/val.parquet` (生成物)

**Acceptance Criteria**:
- `uv run python -m pipeline.case3.training.preprocess` が完走し、`data/lake/case3/{train,val}.parquet` を生成。
- parquet 行数 > 30,000 (最低ライン)。train:val ≈ 9:1。
- 再実行で idempotent (同じ出力)。

---

### Step 3: 特徴量抽出 featurizer.py

**Target**: policy
**Dependencies**: Step 1

**概要**: obs (dict) → BatchFeatures (torch.Tensor) の純粋関数。preprocess.py と推論 agent.py の両方が使う。

**Work Items**:
- [ ] `pipeline/case3/policy/featurizer.py`:
  - `PLANET_FEAT_DIM = 11`, `GLOBAL_FEAT_DIM = 6`, `MAX_PLANETS = 36` を定義。
  - `featurize(obs: dict) -> BatchFeatures` (batch_size=1 の dataclass)。
  - boardSize=100.0 で x/y 正規化、ships/production は log1p、owner は one-hot。
  - fleet を planet 単位に集計 (friendly/enemy/neutral の 3 ch)。
- [ ] `pipeline/case3/policy/types.py`: `BatchFeatures`, `PolicyOutput`, `WorldSnapshot` を frozen dataclass で定義。

**Target Files (Expected)**:
- `pipeline/case3/policy/featurizer.py`
- `pipeline/case3/policy/types.py`

**Acceptance Criteria**:
- Unit test: 合成 obs を渡して `featurize()` が期待 shape `(36, 11)` の tensor を返す。
- 全ての出力値が finite (no NaN/Inf)。

---

### Step 4: Dataset ラッパー

**Target**: training
**Dependencies**: Step 2, Step 3

**概要**: parquet → torch Dataset/DataLoader で学習に流せる形にする。

**Work Items**:
- [ ] `pipeline/case3/training/dataset.py`:
  - `CaseThreeDataset(Dataset[Sample])`, `collate(list[Sample]) -> BatchedSample`.
  - parquet を polars で読み込み、`__getitem__` で行を Tensor に変換。
  - no-op フレーム用の target_label=36 (virtual slot) を扱う。
- [ ] Unit test: fixture parquet (10 行) で DataLoader が期待 shape のバッチを返す。

**Target Files (Expected)**:
- `pipeline/case3/training/dataset.py`
- `tests/pipeline/case3/test_dataset.py`

**Acceptance Criteria**:
- `pytest tests/pipeline/case3/test_dataset.py` PASS。
- Batch shape: planet_feats=(B, 36, 11), from_label=(B,), target_label=(B,), ships_label=(B,)。

---

## Phase 2: モデルと学習 (Step 5-7)

### Step 5: DeepSets モデル

**Target**: policy
**Dependencies**: Step 3

**概要**: PyTorch で 3 ヘッド分類の DeepSets ネットワークを実装。

**Work Items**:
- [ ] `pipeline/case3/policy/model.py`:
  - `ModelConfig(hidden=64, layers=2, ships_buckets=5)` dataclass。
  - `DeepSetsPolicy(nn.Module)`: phi (per-planet MLP) → masked mean pool → psi (global) → 3 heads (from/target/ships)。
  - `forward(BatchFeatures) -> PolicyOutput`。
  - 入力マスクを論理的に適用 (padding は -∞ logit)。
- [ ] Unit test: dummy 入力 (B=2, P=36) で 3 ヘッド shape assert。

**Target Files (Expected)**:
- `pipeline/case3/policy/model.py`
- `tests/pipeline/case3/test_model.py`

**Acceptance Criteria**:
- Parameter 数 < 100K, model size < 1MB (torch.save 後)。
- `test_model.py` で forward shape + マスク効果を assert。

---

### Step 6: 学習ループ train.py + losses.py

**Target**: training
**Dependencies**: Step 4, Step 5

**概要**: BC 学習ループ。AdamW + 3 ヘッド CE loss (1:1:0.5) + ベスト val 重み保存。

**Work Items**:
- [ ] `pipeline/case3/training/losses.py`: `compute_loss(output, labels, weights)`。
- [ ] `pipeline/case3/training/train.py`:
  - config から model/optimizer/scheduler/dataloader を組み立て。
  - エポック毎に train loss + val loss + 3 ヘッド top-1 accuracy を stdout に構造化ログ出力 (JSON 1 行/epoch)。
  - ベスト val loss で `pipeline/case3/policy/weights.pt` に state_dict 保存。
  - seed 固定 (torch/numpy/random, DataLoader worker_init_fn)。
- [ ] CLI: `uv run python -m pipeline.case3.training.train --config configs/il_baseline.yaml`。

**Target Files (Expected)**:
- `pipeline/case3/training/losses.py`
- `pipeline/case3/training/train.py`
- `pipeline/case3/policy/weights.pt` (生成物)

**Acceptance Criteria**:
- 学習が完走 (epochs ~10)、val loss が単調減少する (または最小値で early stop)。
- from_head top-1 accuracy ≥ 30%、target_head top-1 ≥ 20% (検証セット)。
- `weights.pt` サイズ < 1MB。

---

### Step 7: 学習再現性・決定性テスト

**Target**: training / testing
**Dependencies**: Step 6

**概要**: seed 固定で 2 回学習を回して val loss の完全一致を確認。

**Work Items**:
- [ ] `tests/pipeline/case3/test_training_determinism.py` (slow マーカー): 小さな fixture で 2-epoch 学習を 2 回走らせ `val_loss` 完全一致を assert。
- [ ] `pytest -m "not slow"` のデフォルトレーンからは除外。

**Target Files (Expected)**:
- `tests/pipeline/case3/test_training_determinism.py`

**Acceptance Criteria**:
- `uv run pytest tests/pipeline/case3/test_training_determinism.py` PASS。
- fixture は 1,000 行程度の mini parquet で秒単位で終わる。

---

## Phase 3: 推論と評価 (Step 8-11)

### Step 8: Decoder + geometry コピー

**Target**: policy
**Dependencies**: Step 3

**概要**: model 出力 → action list 復元。`aim_with_prediction()` を case3 内に独立コピー。

**Work Items**:
- [ ] `pipeline/case3/policy/geometry.py`: `aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids) -> (angle, ...) | None` を `pipeline/case1/baseline/core/physics.py` から **独立コピー** (ライセンス表記付き)。
- [ ] `pipeline/case3/policy/decoder.py`: `decode(output, world) -> list[list[int|float]]`。
  - 各 my_planet について from_prob 閾値 > 0.5、target argmax, ships bucket argmax の順に greedy 選択。
  - 有効ターゲットマスクで不正手を除外 (自陣 from を from に使う / 到達不可能なら除外)。
- [ ] Unit test: dummy output から期待 action list を assert。

**Target Files (Expected)**:
- `pipeline/case3/policy/geometry.py`
- `pipeline/case3/policy/decoder.py`
- `tests/pipeline/case3/test_decoder.py`

**Acceptance Criteria**:
- `test_decoder.py` PASS (マスク・閾値・no-op パスの 3 ケース)。
- decoder は pure function (副作用なし、同入力で同出力)。

---

### Step 9: Agent エントリポイント + main.py

**Target**: policy
**Dependencies**: Step 5, Step 8, Step 6 (weights.pt)

**概要**: `agent(obs)` の実装と Kaggle 提出 main.py の 20 行ラッパー。

**Work Items**:
- [ ] `pipeline/case3/policy/agent.py`:
  - モジュールロード時に `weights.pt` を 1 回だけロード (`_MODEL = load_model()`)。
  - `agent(obs) -> list[list[int|float]]`: featurize → model(torch.no_grad) → decode。
  - 推論時のタイマーは外部から計測される (src/env/executor.py が wrap)。
- [ ] `pipeline/case3/main.py`:
  - `sys.path.insert(0, str(Path.cwd()))` → `from policy.agent import agent`。
  - 20 行以内。
- [ ] `pipeline/case3/policy/__init__.py` は相対 import で `agent` を公開。

**Target Files (Expected)**:
- `pipeline/case3/policy/agent.py`
- `pipeline/case3/main.py`
- `pipeline/case3/policy/__init__.py`

**Acceptance Criteria**:
- `uv run python -c "from pipeline.case3.policy.agent import agent; print(agent)"` で import 成功。
- `uv run python -c "from kaggle_environments import make; e=make('orbit_wars', configuration={'agents':2,'seed':0}); from pipeline.case3.policy.agent import agent; e.run([agent, agent])"` が例外なく完走。

---

### Step 10: Integration + Snapshot テスト + agent_registry 追加

**Target**: testing / cross-cutting
**Dependencies**: Step 9

**概要**: snapshot 決定性テスト (case1 パターン踏襲) + src/env への登録 + env.run 通し。

**Work Items**:
- [ ] `src/env/agents.py` の `AGENT_REGISTRY` に `"case3_il_v1": "pipeline.case3.policy.agent:agent"` を追加。
- [ ] `tests/pipeline/case3/snapshots/` に fixture obs JSON を 1 件配置。
- [ ] `tests/pipeline/case3/test_agent_snapshot.py`: 同一 obs に対して `agent(obs)` の action JSON が期待値と完全一致することを assert。
- [ ] `tests/pipeline/case3/test_agent_integration.py`: `env.run([case3 agent, baseline v1])` が例外なく完走し、action が legal (from ∈ my_planets, angle ∈ [-π, π], ships > 0) であることを assert。

**Target Files (Expected)**:
- `src/env/agents.py`
- `tests/pipeline/case3/snapshots/obs_*.json`, `action_*.json`
- `tests/pipeline/case3/test_agent_snapshot.py`
- `tests/pipeline/case3/test_agent_integration.py`

**Acceptance Criteria**:
- 全 `tests/pipeline/case3/` PASS (slow を含む)。
- `uv run python -m env run --agents case3_il_v1,case3_il_v1 --mode 1v1 -n 1 --seed 0` が完走。

---

### Step 11: vs baseline 評価スクリプト + 合格基準確認

**Target**: evaluation
**Dependencies**: Step 10

**概要**: case3 vs case1_baseline_v1 を 100 戦走らせ、勝率・draw 率・推論 p95 を出力。

**Work Items**:
- [ ] `pipeline/case3/evaluation/eval_vs_baseline.py`:
  - `src/env/runner.run_episodes(RunSpec(agents=("case3_il_v1","baseline_v1"), mode="1v1", episodes=100, seed=0, parallel=4, save_replay=False, data_root=Path("data")))` を呼ぶ。
  - 出力: 勝率, draw 率, 平均ターン数, 推論 turn_p95 / max, timeouts 数。
- [ ] 合格基準 (要件 §評価ターゲット) を満たすか確認:
  - vs case1_baseline_v1 勝率 ≥ 50%
  - 推論 turn_p95 < 1.0s
- [ ] `pipeline/case3/README.md` に結果を追記。
- [ ] `dev/test-backend` フルレーン (ruff + mypy + pytest) PASS 確認。

**Target Files (Expected)**:
- `pipeline/case3/evaluation/eval_vs_baseline.py`
- `pipeline/case3/README.md` (更新)

**Acceptance Criteria**:
- `uv run python -m pipeline.case3.evaluation.eval_vs_baseline` が完走し、勝率を stdout に出力。
- 勝率 ≥ 50% (下回る場合は Step 6 に戻り HP チューニング or データ拡張)。
- `dev/test-backend` 全レーン PASS。

---

## 並列化可能性

| Parallel 候補 | Step | 備考 |
|---|---|---|
| Step 3 (featurizer) と Step 2 (preprocess) の初期フレーム | どちらも featurizer を使うが、Step 3 を先に完了させてから Step 2 を実装すると手戻り少 |
| Step 5 (model) と Step 4 (dataset) | 双方独立に書ける |
| Step 7 (determinism test) と Step 8 (decoder) | 学習完了後に並列に着手可能 |

基本は逐次進行。1 PR = 1 Phase (3 PR) または 1 PR = 1 Step (11 PR) のどちらかで運用。

## Cross-cutting Concerns

- **命名規約**: すべて `case3_il_v1` を version suffix とする (`case3_il_v2` は後続の改良版)。
- **ロギング**: すべて `logging.getLogger(__name__)` で JSON 構造化。`print()` 禁止。
- **Git コミット**: 各 Step 1 PR、コミットメッセージは `:sparkles:` (新機能) / `:white_check_mark:` (テスト) / `:memo:` (docs) の emoji prefix。
- **ドキュメント**: 各 Step 完了時に `pipeline/case3/README.md` の該当セクションを更新。
