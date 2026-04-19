# Imitation Learning Baseline (case3) — Codebase Research

## 調査目的

`pipeline/case3/` に模倣学習 (Imitation Learning, IL) ベースの提出エージェントを新設する。教師データは `data/lake/` に集約された過去ログ (現状は `data/kaggle_episodes/` に Kaggle 上位陣のリプレイ 800 本弱が蓄積) から抽出する。本章は実装に着手する前に、再利用可能な既存資産・I/F・制約を洗い出す。

## Deep Codebase Analysis

### 1. データ層 — `data/lake/` と `data/kaggle_episodes/`

- **物理パス**: `data/` は worktree 直下から `/Users/user/project/orbit-wars/data` への symlink (mainリポジトリと共有, [`commit 57bd890`])。
- **`data/lake/`**: 現時点では空ディレクトリのみ存在。今後 `case3` の学習用に整形済み (parquet 化された obs/action ペア) を置く想定。
- **`data/kaggle_episodes/matches/`** (実体ある教師データソース):
  - `index.parquet/mode={1v1,ffa4}/run_kaggle_*.parquet` — 試合メタデータ (Hive partition)。
    - 1v1: **456 episodes / 196 unique submissions**。
    - ffa4: **342 episodes**。
    - 主要列: `match_id, mode, seed, turns, winner, draw, agent_{i}_name, agent_{i}_submission_id, agent_{i}_team_id, agent_{i}_rating_mu, agent_{i}_rating_sigma, agent_{i}_state, source, episode_id`。
    - `rating_mu` 範囲: 432〜1488 (mean 887)。**LB 上位 (mu > 1200) でフィルタすれば質の高いデモのみ抽出可能**。
    - `winner` 分布 (1v1): 0=225, 1=223, draw=8 → ほぼフェア、勝者側を expert として学習可能。
  - `replays/kaggle_ep_{id}.json.gz` — 完全リプレイ。1ファイル ~100KB〜700KB、合計 ~300MB 程度。
    - 構造: `{configuration, steps: [[agent0_state, agent1_state, ...], ...], rewards, statuses, info}`。
    - `steps[t][i].observation` = エージェント `i` がターン `t` で見た obs (Orbit Wars そのもののスキーマ)。
    - `steps[t][i].action` = そのターンに `i` が出した action (`[[from_planet_id, angle, num_ships], ...]` のリスト)。
    - `obs` keys: `angular_velocity, comet_planet_ids, comets, fleets, initial_planets, next_fleet_id, planets, player, remainingOverageTime, step`。
    - **典型的な episode は 500 step**。各 step で player 4 人分の obs/action が記録 (1v1 でも内部は 4 席, 余席は inactive)。

### 2. 観測スキーマと特徴量 — `pipeline/case1/baseline/core/`

- `core/types.py:12` `Planet = NamedTuple(id, owner, x, y, radius, ships, production)`
- `core/types.py:22` `Fleet = NamedTuple(id, owner, x, y, angle, from_planet_id, ships)`
- `core/world_model.py:326` `WorldModel(player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids)`:
  - 自陣/敵陣/中立惑星の分類、`my_total/enemy_total/my_prod/enemy_prod` の集計。
  - `arrivals_by_planet` = 全 fleet の到着先・ETA 推定 (build_arrival_ledger)。
  - `base_timeline` = 各惑星について `simulate_planet_timeline(horizon=HORIZON)` の事前計算。
  - `reserve / available / doomed_candidates / threatened_candidates` = 守備バッファの自動計算。
  - `indirect_wealth_map` = 周辺の生産力評価。
- `core/physics.py` (推定): `aim_with_prediction(src, target, ships, ...)` — 相手惑星の未来位置を予測した発射角を返す。**模倣学習では action の `angle` をそのまま回帰するか、target_planet_id を分類するかの設計分岐に効く** (角度はターゲット選定の従属変数)。
- `core/config.py` (定数): `HORIZON, OPENING_TURN_LIMIT, EARLY_TURN_LIMIT` 等の戦略定数。

**再利用方針**: `WorldModel` は重い (1ターン数msだが 500ターン × 800ファイル = 数百万ステップで重要)。学習データ前処理時にも使うが、推論時は薄い特徴抽出関数のみに分離するか検討の余地。

### 3. エージェント I/F と Kaggle 提出基盤

- **エントリポイント規約** (`.claude/rules/pipeline.md`):
  - `pipeline/case<N>/main.py` は `agent(obs)` を公開する 20 行程度の薄い wrapper。
  - `sys.path.insert(0, str(Path.cwd()))` で `Path.cwd()` 経由で `baseline/` をトップレベル import。
  - `__file__` 使用は禁止 (Kaggle Validation で fail)。
  - サブパッケージ内部は **相対 import 必須**。
- **モデル重みの同梱**: `*.pt` / `*.pkl` は packager でバンドル可 (rules:「モデル重み `.pt` / `.pkl` ○同梱」)。`pipeline/.submitignore` に書かないこと。
- **Kaggle ランタイムの依存**: 標準で NumPy / SciPy / PyTorch が利用可能 (kaggle-environments runtime)。**ただし重い依存・大きなモデル重みは validation timeout を起こす**ので、軽量モデル + 最小限の依存に絞る。
- **既存 agents 登録** (`src/env/agents.py`): `AGENT_REGISTRY` に `"case3_il_v1": "pipeline.case3.policy.agent:agent"` の追加が必要。
- **タイミング制約**: `actTimeout=1` 秒/ターン。`src/env/executor.py:20` `TIMEOUT_THRESHOLD_SEC = 1.0` で計測される。**推論は明確に 1 秒以下、目安 100ms 以下**。

### 4. case1 (rule-based reinforce baseline) の構造

- `pipeline/case1/baseline/` 配下に `core/`, `missions/`, `strategy.py` (25KB)。**手書きで 25KB の戦略コード**。これを完全に置き換える模倣学習エージェントが case3 のターゲット。
- `evaluation/snapshot_update.py`: 観測/action 固定スナップショットでの決定性検証。**case3 でも IL モデルの推論決定性 (same input → same output) を担保するため同じ仕組みを流用すべき**。
- `configs/baseline.yaml`: 戦略パラメータの YAML 化 (Kaggle へは同梱せず)。

### 5. 評価フレームワーク — `src/env/`

- `src/env/runner.py:115` `run_episodes(spec)` — multiprocessing で N エピソードを回し parquet + replay.json.gz を吐く。
  - **学習後の評価**: `case3_il_v1 vs baseline_v1` を多数回回して勝率比較が標準フロー。
- `src/env/recorder.py:36` `write_records()` — `data_root/matches/index.parquet/mode={mode}/run_{run_id}.parquet` 形式。
- `src/env/types.py` `MatchRecord` (schema v2) — Kaggle/selfplay 両方に対応。
- `src/env/kaggle/scraper.py:169` — 既存の Kaggle episode 取得パイプ。**追加データが必要なら再実行で増やせる**。

### 6. テストと CI

- `tests/pipeline/case1/snapshots/` — 観測/action JSON snapshot で agent の決定性を検証。
- `dev/test-backend` = format check → lint (ruff) → mypy → pytest。**case3 もこの全レーンを通す必要あり**。
- `pyproject.toml` 既存依存: numpy, pandas, polars, pyarrow, pytest, ruff, mypy, pydantic, typer, rich, requests。**PyTorch は未追加**。

## 技術的制約

| 制約 | 詳細 | 影響 |
|------|------|------|
| Kaggle actTimeout | 1.0 秒/ターン | 推論モデルは小さく (<10MB)、torch.no_grad + CPU 推論前提 |
| Submission サイズ | 100MB 上限 (Kaggle 共通) | モデル重み + コードで 100MB 以内 |
| Kaggle ランタイム依存 | torch は CPU 版を期待 (pinned版はノーチェック) | ローカル学習と推論で torch バージョン整合性に注意 |
| 教師データ品質 | rating_mu 432〜1488 と幅広い | 上位 (mu>1200) フィルタ必須 / そうでないと弱い行動も模倣 |
| Action 連続性 | `angle` は連続値 (radian)、`num_ships` は整数 1〜N | 回帰 vs 分類 vs ハイブリッドで設計分岐 |
| Action のリスト長可変 | `[]` (no-op) も合法、複数発射も可能 | 出力ヘッドの設計で「何隻発射する手を何個出すか」を扱う必要 |
| Determinism | snapshot test で固定 | torch.manual_seed + eval mode + greedy 推論で担保 |

## 既存機能でカバーされる部分 / 未対応部分

| 領域 | 既存 | 不足 (case3 で実装) |
|------|------|---------------------|
| 試合メタ取得 | `src/env/kaggle/scraper.py` で完了 | 追加スクレイプは適宜 |
| 観測パース | `pipeline/case1/baseline/core/types.py` の Planet/Fleet 型 | **学習用の特徴量 tensorizer** |
| 行動表現 | env.run が `[[from_id, angle, ships], ...]` を期待 | **モデル出力 → action list の decoder** |
| 並列対戦評価 | `src/env/runner.py` で完備 | `case3_il_v1` を `agents.py` に追加するだけ |
| 模倣学習 | なし | **データセット / モデル / 学習ループ / 推論ラッパ** すべて新規 |
| モデル同梱 | packager が `.pt` 対応 | `pipeline/case3/policy/weights.pt` の配置 |

## Key Findings Summary

- **教師データは即使える**: `data/kaggle_episodes/matches/replays/*.json.gz` × 798 ファイル、parquet メタで rating filter 可。`data/lake/` は空なので、case3 の前処理で `data/lake/case3/episodes.parquet` のように整形済みデータを書き出す方針が自然。
- **観測 → 特徴量変換は既存 `WorldModel` を流用可能**だが、学習時は重さがネック。**最小限の Planet/Fleet テンソル化** (NumPy 配列 → torch.Tensor) を別途用意し、`WorldModel` は ETA や reserve 計算など補助特徴量にだけ使うのが現実的。
- **Action 空間の設計が最大の論点**:
  - (a) 「from_planet_id を分類 / angle を回帰 / ships を回帰」の 3 ヘッド構成
  - (b) 「from_planet × target_planet ペアを分類 / ships は需要式から決定」の組合せ
  - (c) 「(from, target, ships_bucket) の組合せを完全分類 → top-k actions」
- **Kaggle 提出規約は厳密**: `Path.cwd()` ベースの sys.path、相対 import、`.submitignore` メンテ — case1 と同じパターンを踏襲する。
- **PyTorch を新規依存に追加する必要**。`pyproject.toml` への追加と `dev/setup` への影響を Step 5 (Architecture) で検討。
- **既存 case1 (rule-based) を強い対戦相手として使える**。case3 の評価指標は「case1_baseline_v1 vs case3_il_v1 の勝率」で素直に出せる。
