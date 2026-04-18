# Evaluation System — Web Technical Research

## 1. Official Documentation

### 1.1 `kaggle_environments` — コアAPI

`kaggle_environments/core.py` の読解（L154-821）により以下を確定:

- **`make(name, configuration=None, info={}, steps=[], logs=[], debug=False)`**
  - `steps` 引数にリストを渡すとリプレイを復元可能。`__set_state(steps[-1])` を呼んで最終状態から巻き戻す。
  - `configuration={"agents": N, "seed": s, "episodeSteps": 500, ...}` で対戦条件を指定。
  - **重要**: `seed` を config に含めるだけでは「完全決定性」は保証されない（docstring / snapshot_update.py の注記どおり）。
- **`env.run(agents)` (L540-569)**: agent 関数のリストを受け、`done` か runTimeout まで自動で step。内部で `self.pool = Pool(processes=len(agents))` を**1回だけ**確保（L810-821、`agent.is_parallelizable` が True の場合のみ）。
  - 注意: Pool は env に保持されるので、同一 env オブジェクトを使い回す場合は明示解放が必要。通常は use-and-discard で OK。
- **`env.step(actions)`**: 手動1ターン進行。
- **`env.toJSON()` (L669-688)** が返すフィールド:
  ```
  id, name, title, description, version, module_version,
  configuration, specification (action/agents/configuration/info/observation/reward),
  steps, rewards, statuses, schema_version, info
  ```
  これを `json.dumps(env.toJSON())` で保存 → `json.loads()` 後 `make(name, configuration=..., steps=loaded["steps"])` で復元可能。
- **`env.render(mode=...)`** の4モード:
  | mode | 説明 | 用途 |
  |---|---|---|
  | `html` | 自己完結HTML文字列を返す | 保存・ブラウザで開く |
  | `ipython` | IPython の `display(HTML(...))` で直接描画 | Jupyter 上で再生 |
  | `json` | `json.dumps(toJSON())` | 分析・シリアライズ |
  | `ansi` / `human` | テキスト / stdout | デバッグ |
- **Orbit Wars envの構成** (`kaggle_environments/envs/orbit_wars/`):
  - `orbit_wars.py` — 中核エンジン（step, observation, 戦闘解決）
  - `orbit_wars.js` — ブラウザ描画用 JS（`html` / `ipython` モードの実体）
  - `orbit_wars.json` — specification（デフォルト config）
  - `visualizer/default/` — 描画用アセット

### 1.2 Polars — Parquet I/O

- `DataFrame.write_parquet(path, partition_by=["mode", "date"])` で Hive 形式の分割書き込み可能 (`pola.rs/user-guide/io/hive`)。
- 追記は「新しいパーティションを書く」方式が公式推奨（Issue #18750）。同一パーティション内での append は未サポート → **run_id 列を付与し、run_id パーティションで書けば衝突しない**。
- `pl.scan_parquet("data/matches/**", hive_partitioning=True)` で遅延読み込み・集計が可能。

### 1.3 Python `multiprocessing.Pool`

- CPU-bound で GIL の影響を受けない。`Pool(processes=os.cpu_count())` が目安、メモリ制約があれば半減。
- `pool.map(run_episode, seeds)` で並列エピソード実行。
- **kaggle_environments 由来の注意**: `env.run()` 内部で子プールを起こすが、`is_parallelizable` が False の agent（ローカル関数）ではプール作成されない。外側で親プロセスが multiprocessing.Pool を使う場合、子プロセス内で env を make しなおすのが安全（pickling の問題）。

## 2. Similar OSS Projects

### 2.1 google-deepmind/open_spiel
- **Relevance**: マルチエージェント強化学習の研究向けフレームワーク。トーナメント評価パターンが参考になる。
- **Approach**:
  - round-robin pairwise で全 agent 総当たり。`ranking.csv` と `results.csv` の2ファイル出力。
  - Python 実装は 1 actor = 1 process、C++ 実装はスレッド + バッチ推論。
- **Reusable patterns**:
  - 出力を「ranking（集約）」と「results（1戦1行）」の2レイヤに分けるのは `index.parquet` + `matches/*.parquet` 設計と親和性が高い。
- **Pitfalls**: Python 実装は推論が CPU 限定で遅い。本件は推論ではなくシミュレーション律速なので、この制約は当てはまらない。

### 2.2 vivekjoshy/openskill.py
- **Relevance**: 本件でレーティング計算を入れる場合の候補。TrueSkill より軽量・Python純。
- **Approach**: `Model().rate(teams)` で N(μ, σ) を更新。Kaggle のスキルレーティングと類似の正規分布モデル。
- **Reusable patterns**: Kaggle 側は勝敗のみで更新するため、**ローカル評価でも同じルールに従って `openskill` を回せば、Kaggle 投稿前に μ 期待値を予測できる**。
- **Pitfalls**: なし（軽量ライブラリ）。ただし MVP では「勝率」で十分、openskill は second-iteration に回す判断がある。

### 2.3 probberechts/ML_project20
- **Relevance**: OpenSpiel 向けのトーナメントランナー（1大学課題）。ミニマル実装の参考。
- **Reusable patterns**: agent を `{name: callable}` マップで受け取り、全ペアを生成→実行、結果を CSV 蓄積。本件の agent_registry の雛形に近い。
- **Pitfalls**: 並列化が `concurrent.futures` で粗い。プロセス再利用がない。本件は `multiprocessing.Pool` で改善。

### Pattern Comparison

| Aspect | Our Project | OpenSpiel | ML_project20 |
|---|---|---|---|
| Agent registry | 新規 `src/env/agents.py` で `{name: callable}` | dict of bots | `agents = {"a": fn_a, ...}` |
| Match scheduling | 新規 (pair × seed matrix) | round-robin | 全組合せ |
| Output | Parquet (match) + JSON (replay) | CSV x2 | CSV |
| Replay viewer | Jupyter `env.render("ipython")` | なし | なし |
| Parallel | `multiprocessing.Pool` | 1 actor = 1 process | `concurrent.futures` |

## 3. Library/Service Selection

### 3.1 リプレイ保存形式

| 候補 | Pros | Cons | Maintenance | Recommendation |
|---|---|---|---|---|
| JSON (env.toJSON()) 単体 | 復元容易、標準 | サイズ大、集計に不向き | — | サブセットとして保持 |
| ⭐Parquet(match summary) + JSON(replay) 併用 | 集計高速・再生可 | 二層管理 | Polars安定 | **推奨** |
| JSONL のみ | 追記容易 | 集計遅、スキーマなし | — | 却下 |
| SQLite | クエリ容易 | ファイル肥大・並列書き込み弱 | — | 却下 |

💡 推薦理由: 分析は Polars + Parquet がプロジェクト標準（pyproject に `polars>=1.39` / `pyarrow>=23` 既に導入済み）。再生は kaggle_environments の `make(..., steps=...)` を使うため、steps だけは JSON で残す二層構成が最も低コスト。

### 3.2 可視化ライブラリ

| 候補 | Pros | Cons | Maintenance | Recommendation |
|---|---|---|---|---|
| ⭐kaggle_environments 標準 `env.render` | 実装0、公式同一、`ipython`/`html` | Jupyter 必須 | 公式 | **推奨** |
| Streamlit / Dash | インタラクティブ | 依存重・保守増 | 活発 | 却下（要件外） |
| matplotlib/plotly で自作 | カスタム | 労力大 | 活発 | 却下 |

💡 推薦理由: ユーザー要件が「ノートブックで `env.render()` を呼ぶだけ」。`pipeline/case1/eda/` 以下に可視化用ノートブック（.py）を1つ置けば完了。

### 3.3 並列化方式

| 候補 | Pros | Cons | Recommendation |
|---|---|---|---|
| ⭐`multiprocessing.Pool` | CPU並列、標準ライブラリ | プロセス起動コスト | **推奨** |
| `concurrent.futures.ProcessPoolExecutor` | Pool と同等、futures API | API 違いだけ | 却下 |
| `asyncio.gather` | 軽量 | CPU bound では無効 | 却下 |
| Ray | スケール容易 | 依存大、本件では過剰 | 却下 |

💡 推薦理由: pipeline.md でも指定。依存追加なし。

### 3.4 レーティング（オプション）

| 候補 | Pros | Cons | Recommendation |
|---|---|---|---|
| 勝率のみ（MVP） | シンプル・明確 | 対戦相手強度を考慮できない | **MVP 推奨** |
| openskill.py | Kaggleに近い正規分布ベース | 依存追加 | Phase2 推奨 |
| trueskill | 実績豊富 | Microsoft ライセンス注意 | 却下 |
| elo | 簡単 | 2人専用 | 却下 |

💡 推薦理由: MVP は勝率のみで出し、後続イテレーションで openskill を追加する。

## 4. API/Protocol Research

- **Orbit Wars の observation dict スキーマ**（`snapshot_update.py` の動作から確定）:
  ```json
  {
    "player": int,
    "step": int,
    "planets": [[id, owner, x, y, radius, ships, production], ...],
    "fleets": [[id, owner, x, y, angle, from_planet_id, ships], ...],
    "angular_velocity": float,
    "initial_planets": [[...]],
    "comets": [...],
    "comet_planet_ids": [...]
  }
  ```
- **リプレイ復元の最小契約**:
  - 保存: `env.toJSON()` をそのまま JSON で保存。
  - 復元: `loaded = json.loads(...)`, `env = make("orbit_wars", configuration=loaded["configuration"], steps=loaded["steps"])`, `env.render(mode="ipython")`.
- **既知の observation 問題**: discussion「initial_planets differs by player after comet updates」— コメット更新後に `initial_planets` が player ごとに異なる可能性。記録時は player 0 視点固定にせず、全 agent 分 (`env.steps[-1][i]["observation"]`) を保持するか、`env.toJSON()["steps"]` をそのまま丸ごと保持する（後者が安全）。

## 5. Research Summary

### 採用する方針

1. **リプレイ保存**: `env.toJSON()` を `data/matches/replays/{match_id}.json.gz` にgzip保存。サイズ対策に gzip 必須。
2. **集計データ**: 各マッチ1行の集計を `data/matches/index.parquet` に Hive パーティション（`mode=1v1` など）で append。
3. **再生**: ノートブック (`pipeline/case1/eda/replay_viewer.py` — percent-format) で `load_match(match_id)` → `env.render("ipython")`。
4. **並列化**: `multiprocessing.Pool(os.cpu_count())` でエピソードを batch 実行。子プロセス内で env を make。
5. **Agent registry**: `src/env/agents.py` に `{name: callable}` マップ。案: `"baseline_v1" → pipeline.case1.baseline.agent.agent`, `"case0" → pipeline.case0.main.agent`, `"random" → kaggle_environments random`。
6. **CLI**: `src/env/cli.py` を `typer` で実装。`python -m src.env.cli run --agents baseline_v1,case0 --mode 1v1 --episodes 100 --parallel 8`。
7. **レーティング**: MVP は勝率のみ、openskill は Phase2。

### 根拠となる外部ソース

- [Kaggle/kaggle-environments core.py](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py) — toJSON / render / make の仕様。
- [Kaggle/kaggle-environments envs/orbit_wars](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars) — orbit_wars.js 描画、orbit_wars.json spec。
- [Polars Hive partitioning](https://docs.pola.rs/user-guide/io/hive/) — partition_by + scan_parquet(hive_partitioning=True)。
- [Polars parquet append issue #18750](https://github.com/pola-rs/polars/issues/18750) — 新パーティション追記パターン。
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html) — Pool / os.cpu_count 指針。
- [OpenSpiel tournament](https://openspiel.readthedocs.io/) — ranking.csv + results.csv の二層構造。
- [probberechts/ML_project20](https://github.com/probberechts/ML_project20) — agent registry + 全ペア実行ミニマル実装。
- [openskill.py](https://github.com/vivekjoshy/openskill.py) — Phase2 レーティング候補。
