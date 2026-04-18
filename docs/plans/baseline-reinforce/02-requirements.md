# baseline-reinforce — 要件定義

## 背景と目的

Kaggle「Orbit Wars 2026」コンペ参戦プロジェクトにおいて、公開ノートブック **「Orbit Wars 2026 - Reinforce」(sigmaborov, Public Score 928.5, v2, Apache 2.0)** を忠実に再現した **ミッション計画型ヒューリスティックエージェント** を `pipeline/case1/` に構築する。

このベースラインは以下の三つの役割を担う:

1. **リーダーボード提出用エージェント** — そのまま Kaggle に提出し 900+ スコアを確保。
2. **改良の土台となる参照実装** — 以後 `case2, case3, ...` で改良する比較基準。
3. **コードベース設計の検証サンドボックス** — `pipeline/caseN/` 構造と CI パイプラインが実運用に耐えるかを検証。

## ユーザーストーリー

（優先度: P0 必須 / P1 推奨 / P2 余力）

- **US-1 (P0)**: 開発者として、`kaggle kernels pull sigmaborov/orbit-wars-2026-reinforce -p pipeline/case1/notebook/` で元ノートブックを手元に取得できる。
- **US-2 (P0)**: 開発者として、`python -m pipeline.case1.evaluation.selfplay` コマンドで 1v1 / 4P FFA の自己対局を N エピソード実行し、勝率・タイムアウト率・平均ターン数を取得できる。
- **US-3 (P0)**: 開発者として、`pipeline/case1/baseline/main.py` をそのまま Kaggle に `kaggle competitions submit` で提出できる。
- **US-4 (P0)**: 開発者として、CI 相当コマンド（`uv run ruff check pipeline/case1 tests && uv run mypy pipeline/case1 && uv run pytest tests`）が警告・エラー無しで緑になる。
- **US-5 (P1)**: 開発者として、`data/replays/case1/<timestamp>/` に各エピソードのリプレイ JSON と集計 CSV が保存され、後で確認できる。
- **US-6 (P1)**: 開発者として、`pipeline/case1/README.md` を読めば case1 の戦略・実行方法・既知の制約を把握できる。

## 機能要件

### FR-1: ノートブック取得手段の提供
- `pipeline/case1/notebook/` ディレクトリを新規作成する。
- Kaggle API 経由で `sigmaborov/orbit-wars-2026-reinforce` を pull し、`.ipynb` とメタデータ (`kernel-metadata.json`) を同ディレクトリに保存する。
- 取得手順を `pipeline/case1/README.md` に明記し、再取得可能にする。

### FR-2: ノートブック再現エージェント
- `pipeline/case1/baseline/` 配下に、ノートブックの以下のコンポーネントを **種類別ファイル分割**して再現する:
  - `config.py` — 全 CONFIG パラメータ (80+) を Python モジュール定数として配置。ノートブックと同一名・同一値。
  - `types.py` — `Planet`, `Fleet` namedtuple（kaggle_environments と重複するが import safety のため独立定義を保持）。
  - `geometry.py` — `dist`, `segment_hits_sun`, `point_to_segment_distance`。
  - `physics.py` — `fleet_speed`, `orbital_radius`, `is_static_planet`, `travel_time`, `predict_planet_position`, `predict_comet_position`, `predict_target_position`。
  - `world_state.py` — `WorldState` dataclass、`build_arrival_ledger`, `simulate_planet_future`, `projected_state`, `search_safe_intercept`。
  - `missions/expansion.py` — `build_expansion_missions`。
  - `missions/attack.py` — `build_attack_missions`。
  - `missions/snipe.py` — `build_snipe_missions`。
  - `missions/swarm.py` — `build_swarm_missions`（2-source / 3-source 両対応）。
  - `missions/reinforcement.py` — `build_reinforcement_missions`。
  - `missions/crash_exploit.py` — `build_crash_exploit_missions`（4P専用）。
  - `agent.py` — `agent(observation, configuration=None)` を公開。
  - `main.py` — Kaggle Submission エントリポイント（`from .agent import agent` を再エクスポート）。
- 挙動はノートブック v2 と完全一致（同一 seed の 1v1 で同一アクションシーケンスを返すことをスナップショットテストで検証）。

### FR-3: ライセンス表示
- `pipeline/case1/baseline/` のすべてのファイル先頭に Apache 2.0 ライセンス表記と原典表記コメントを配置:
  ```python
  # Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
  # https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-reinforce
  # Licensed under Apache License 2.0
  ```
- `pipeline/case1/baseline/LICENSE` に Apache 2.0 本文を配置。

### FR-4: 自己対局 CLI
- `pipeline/case1/evaluation/selfplay.py` を `typer` CLI として実装する。
- オプション:
  - `--episodes N` (default: 20)
  - `--mode {1v1, ffa4}` (default: 1v1)
  - `--seed SEED` (default: 0)
  - `--output-dir PATH` (default: `data/replays/case1/<timestamp>/`)
  - `--save-replay / --no-save-replay` (default: save)
- 出力:
  - 各エピソードの JSON リプレイ (`episode_<i>.json`)
  - 集計 CSV (`summary.csv`): episode_id, winner, turns, p0_final_score, p1_final_score, timeouts, elapsed_sec
  - `rich` テーブルでサマリを標準出力（勝率・平均ターン・タイムアウト率）

### FR-5: CONFIG の YAML 抽出基盤
- `pipeline/case1/configs/baseline.yaml` にノートブック CONFIG と同じパラメータ群を列挙する（後工程でチューニング時に差し替え可能にする事前配線）。
- 本 feature では Python 定数が優先されるが、`configs/baseline.yaml` の存在と `load_config(path)` ヘルパーのみ提供（実際の差し替えは scope外）。

### FR-6: ビルド・品質
- `pyproject.toml` に以下を追加:
  - `[dependency-groups.dev]` に `"kaggle>=1.7.4"` を追加。
  - `[tool.ruff.lint.per-file-ignores]` に `"pipeline/case1/baseline/**" = ["C901", "E501", "PLR0912", "PLR0913", "PLR0915"]` を追加（ノートブック互換性のため）。
  - 他の `pipeline/case1/**` (evaluation, configs) は既定 strict を守る。
- `pipeline/case1/baseline/` 配下は mypy strict を維持。`Planet | None`、`int`, `float` を厳密に型付け。
- テスト: `tests/pipeline/case1/test_baseline_agent.py` (動作確認＋ snapshot)、`tests/pipeline/case1/test_world_state.py` (単体)。

## 非機能要件

| 項目 | 目標値 | 計測方法 |
|------|--------|----------|
| 1ターン実行時間 | P95 < 1.0s、P99 < 2.0s (overage time 内) | selfplay CLI で全ターン `time.perf_counter` 計測し JSON に記録 |
| 自己対自己 DONE 到達率 | 100 エピソードで 100% | selfplay 終了後の `env.state[0]["status"] == "DONE"` を集計 |
| ノートブック挙動一致 | 同一 seed / 同一 observation に対する action sequence が一致 | スナップショットテスト: `tests/pipeline/case1/snapshots/episode_seed0.json` を生成し、以降差分が出れば失敗 |
| Ruff / Mypy | 本 feature が触る範囲で warnings 0 | `uv run ruff check pipeline/case1 tests && uv run mypy pipeline/case1` |
| Pytest | 追加テストが緑、既存カバレッジ (`--cov=src`) は現状維持 | `uv run pytest tests` |
| リプレイ永続化 | 全エピソードの JSON を `data/replays/case1/<timestamp>/` に保存 | CLI 既定動作、`data/` は `.gitignore` 管理 |

## スコープ外（本 feature で扱わない）

- **src/ への モジュール分割（Step B）**: 次回 feature で `src/agents/`, `src/features/`, `src/policies/`, `src/utils/` へ移行。
- **CONFIG パラメータの探索・チューニング**: 値はノートブックと同値を維持。別 feature で CMA-ES / Optuna 等を使ったチューニングを実施。
- **RL / 模倣学習による強化**: ベースラインの延長線上で別 case (`pipeline/case2/` 以降) として独立させる。
- **外部ボット (Random / NearestStrategy) に対するベンチマーク網羅**: 本 feature は **セルフプレイのみ**。他ボットとの対戦は次 feature の評価基盤で。
- **dev/setup スクリプトの修正**: 旧構造の残骸除去は scope外。本 feature 内では `uv sync` および `uv run` を直接叩く。
- **可視化 (matplotlib / js renderer 連携)**: リプレイ JSON の保存までとし、可視化は別途。

## 用語集

| 用語 | 説明 |
|------|------|
| **Reinforce ノートブック** | sigmaborov 氏の Kaggle notebook v2、Public Score 928.5 |
| **Mission** | エージェントが評価する行動候補。`expand/attack/snipe/swarm/reinforce/crash_exploit` の6種 |
| **WorldState** | 現在ターンの観測から派生状態（`my_planets`, `arrivals_by_planet`, `doomed_planets` 等）を集約する dataclass |
| **Arrival Ledger** | 各惑星への飛行中フリート到着予定を `{planet_id: [(eta, owner, ships), ...]}` で保持 |
| **Projected State** | コミット済み攻撃を加味した未来の惑星状態投影（`base_need_cache` で高速化） |
| **Doomed planet** | 現時点の arrivals から算出した結果、防衛不能と判定される自軍星 |
| **Crash-Exploit** | 4P 戦で敵フリートが太陽で自壊する直後を狙ってキャプチャするミッション |
| **Swarm (2/3-source)** | 複数の自軍星から同一敵星へほぼ同時着弾させる集中攻撃 |
| **Selfplay** | 同一エージェントを 2〜4 プレイヤーに割り当てて回す自己対局 |
