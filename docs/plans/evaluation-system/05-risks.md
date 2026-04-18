# Evaluation System — Risks and Dependencies

## 1. Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|---|---|---|---|
| 1 | **ストレージ肥大化（JSONのサイズ爆発）** — N=1000 で save-replay=True だと 1-5GB 消費、gitignore 漏れで push 事故も | 高 | 高 | `--save-replay` を opt-in（デフォルト False）。`.gitignore` に `data/matches/**`・`data/replays/**` を先に追加。`env prune --older-than <D>d` を Phase2 で提供 |
| 2 | **Orbit Wars 環境の非決定性**（seed 同一でもエピソード長が変わる） | 中 | 高 | integration テストでは「勝者」「長さ」をアサートしない。`return type`/`レコード件数`/`JSON round-trip` など seed 非依存の性質のみ検証 |
| 3 | **multiprocessing での pickle 失敗** — agent がクロージャ / ローカル関数だと子プロセスへ渡せない | 高 | 中 | agent は `"module:attr"` 文字列のみ保持し、ワーカー内で lazy import。タイミングラッパーも top-level 関数として提供 |
| 4 | **kaggle_environments のバージョン不整合** — PyPI 1.17.0 には orbit_wars が含まれない可能性（docs/competition/20260418_evaluation.md） | 高 | 中 | `pyproject.toml` で `kaggle-environments>=1.28.0` へバンプ要否を Step 0 で確認。CI の `uv sync` で make("orbit_wars") が成功するかスモークテスト |
| 5 | **Polars hive partitioning のファイル衝突** — 同 run_id の同時書き込み | 中 | 低 | `recorder.write_run` は run 終了時にメインプロセスで 1 回だけ書く。並列書き込み禁止。ファイル名は `run_{ULID}_{mode}.parquet` でユニーク |
| 6 | **Kaggle submission 事故** — 評価実行中に誤って Kaggle へ提出 | 致命 | 極低 | `src/env/` は `src/submit/` を一切 import しない。CLAUDE.md の Kaggle Submission Policy を遵守。評価 CLI (`env`) は submission コマンドを提供しない |
| 7 | **`dev/test-backend` の既存バグ** — `cd backend` が存在せず CI が機能不全の可能性 | 中 | 中 | Step 0 で修正。ルート実行に変更して既存テストが通ることを事前確認 |
| 8 | **integration テストのフラキー化** — kaggle_environments が稀にタイムアウト扱いになる | 低 | 低 | `@pytest.mark.integration` マーカーと `pytest.importorskip("kaggle_environments")` を活用。CI では integration を optional ジョブに分離（Phase2） |
| 9 | **Parquet スキーマ drift** — player 数が将来変わったときに 4 固定列が合わなくなる | 低 | 低 | `MatchRecord.to_row()` に schema version 列を入れる。2人/4人以外は現仕様で未対応として明示 |
| 10 | **リプレイロードの OOM** — 大きい JSON を全部オンメモリに読む | 低 | 低 | 1 ファイル = 1 エピソード、通常 1-5MB 以下。同時多数ロードは想定外だが、必要なら streaming 読み出しを Phase2 |

## 2. External Dependencies

- **kaggle-environments** — `orbit_wars` env 含有バージョンが必須。`>=1.28.0` を目安に pyproject で指定。
- **polars / pyarrow** — `polars>=1.39`、`pyarrow>=23.0.1`（既存）。hive partitioning の正式サポート版。
- **rich / typer** — 既存依存。追加不要。
- **multiprocessing** — 標準ライブラリ。macOS のデフォルト start method は `spawn`、Linux は `fork`。本件では明示的に `get_context("spawn")` を使って両環境で揃える。
- **git** — `agent_version()` で `git rev-parse --short HEAD` を呼ぶ。git 環境が無い場合は `""` を返してフェイルしないこと。

## 3. Technical Debt

- `pipeline/case1/evaluation/snapshot_update.py` は責務が別物（テスト用スナップショット生成）だが残置する → 後に `tests/pipeline/case1/_fixtures/` 配下へ移動する技術債となる可能性。
- `MatchRecord` を 4 人固定列で持つため、将来 N 人対戦が追加されると大改修が必要。現時点の Kaggle コンペでは 1v1 / FFA4 のみなので容認。
- `agent_version()` は `git rev-parse` を呼ぶが、uncommitted 変更がある場合の扱いを記録していない。Phase2 で `git status --porcelain` 結果を suffix に付ける（例: `ab12cd3-dirty`）。
- `openskill` によるレーティング計算は未実装（要件で Phase2 に延期）。

## 4. Open Items

- [ ] kaggle-environments のバージョン実測 — `uv run python -c "from kaggle_environments import make; make('orbit_wars')"` を Step 0 の事前確認で実施する
- [ ] `multiprocessing` の並列数デフォルト値（`os.cpu_count()` か固定値 4 か）— Step 5 実装時に決定
- [ ] `match_id` 形式の最終決定（ULID vs `{ts}_{sha}_{seed}`）— Step 1 実装時に決定。ULID 導入なら依存追加必要
- [ ] CI で integration テストを走らせるかどうか — 現在 ci-backend.yml 未確認。Step 11 で判断

## 5. 分離ポリシー（Kaggle submission）

**最重要の安全弁**として、`src/env/` は `src/submit/` から完全独立とする:

- `src/env/**` は `src.submit` を import しない（コードレビューで確認）
- `env` CLI は `submit` / `archive` / `upload` 系サブコマンドを提供しない
- テストで `src.submit` を触らない（既存 `tests/submit/**` は変更しない）
- CI で `grep -r "src.submit" src/env/ && exit 1` のような guard を検討（Phase2）

この分離により、評価実行中に Kaggle 提出クォータを誤消費するリスクをアーキテクチャ的にゼロにする。
