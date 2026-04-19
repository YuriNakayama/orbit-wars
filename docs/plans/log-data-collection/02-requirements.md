# Log Data Collection — Requirements Definition

## 背景と目的

Kaggle Orbit Wars の **リーダーボード上位エージェントが実際にプレイしたエピソードのログ（リプレイ JSON）** を継続的に収集し、**模倣学習用データセット** として利用する。自己対戦だけでは到達できない高水準の戦略分布を、behavior cloning や RL pretraining の教師データとして活用する。

**主用途**: 観測 → 行動の (s, a) ペア抽出、勝者側の意思決定パターン学習、希少イベント（コメット出現時の対応、終盤詰め寄り）のサンプル強化。

副用途として、自作エージェントの比較ベンチマーク（上位エージェントとの対戦結果との一致率）にも利用可能とする。

## User Stories

- **As a 自作エージェント開発者**, I want リーダーボード上位のリプレイを一括取得したい, so that 模倣学習用の高品質データセットを作成できる。
- **As a 戦略分析者**, I want mode（1v1/ffa4）別に最新の上位対戦ログを手元に同期したい, so that 上位プレイヤーの戦術トレンドを追跡できる。
- **As a パイプライン運用者**, I want 再実行しても既取得エピソードは skip し増分だけ取得したい, so that 帯域と API クォータを浪費しない。
- **As a 開発者**, I want Dry-run でスコープを確認してから実 DL したい, so that 意図しない大量 DL を防げる。
- **As a 研究者**, I want 取得したリプレイを既存の `loader.load_replay()` / `env.render()` でそのまま再生したい, so that 可視化・分析コードを書き直さずに済む。

## Functional Requirements

### FR-1 リーダーボード取得
- `kaggle competitions leaderboard -c orbit-wars --show` の CSV stdout を parse して `(rank, team_id, team_name, score, submission_date)` を構造化。
- `--top N` オプションで上位 N team を取得（既定値: 30）。

### FR-2 Episode 一覧取得
- 各 team_id について `list_episodes_for_team(team_id)` を呼び、episode メタデータを取得。
- 既取得 `episode_id` は Parquet index を参照して skip。
- `--modes 1v1,ffa4` 指定で mode フィルタ（agents 数で判定: 2 → 1v1, 4 → ffa4）。

### FR-3 リプレイ DL と保存
- 未取得の episode について `get_episode_replay(episode_id)` を呼び、`result.replay` 文字列を gzip 圧縮して `data/kaggle_episodes/replays/{episode_id}.json.gz` に保存。
- 保存後、メタ情報を `MatchRecord` 互換形式で Parquet index に append。
- TIMEOUT/ERROR/INVALID 状態の episode は replay 欠損が多いため、既定で skip（`--include-failed` で取得可）。

### FR-4 レジューム
- 既存 `data/kaggle_episodes/matches/index.parquet/mode=.../**/*.parquet` から `episode_id` を unique 抽出。
- 同一 run 内の重複も in-memory set で抑止。

### FR-5 Dry-run
- `--dry-run` で API 呼び出しは行うが replay DL と書き込みは実行せず、収集対象件数と推定サイズのみ stdout に表示。

### FR-6 CLI
- 新規モジュール `src/env/kaggle/cli.py` の typer app。サブコマンド:
  - `scrape` — 一次ルート実行（上位 N team → episodes → replays）。
  - `list` — 収集済みリプレイの index 表示（既存 `env list` 相当の Kaggle 版）。
  - `inspect` — 単一 `episode_id` のメタ情報表示。

### FR-7 メタデータスキーマ拡張
- `MatchRecord` の `SCHEMA_VERSION` を 2 に更新。以下を追加:
  - `source: str`（`"kaggle"` / `"selfplay"`）
  - `episode_id: int` (nullable, selfplay は 0 または -1)
  - `scraped_at: str` (UTC ISO 8601)
  - 各 agent に `agent_{i}_submission_id`, `agent_{i}_team_id`, `agent_{i}_rating_mu`, `agent_{i}_rating_sigma`, `agent_{i}_state`
- 既存 selfplay レコードとの互換性は確保（新列はデフォルト値で埋める）。

### FR-8 エラーハンドリング
- HTTP 429 → `urllib3.Retry` の backoff で対応、最大 5 回。
- ネットワーク例外 → 該当 episode を skip、次へ進む。スクレイピング全体は止めない。
- `KeyboardInterrupt` → その時点までの run を Parquet に書き出して正常終了。

## Non-Functional Requirements

### NFR-1 レートリミット
- Kaggle EpisodeService への呼び出しを **60 req/min 以下** に抑える自前トークンバケット。
- バースト発生時は `time.sleep` で平滑化。

### NFR-2 再現性
- CLI 実行は決定的（`--top`, `--modes`, `--run-id` 指定で同一結果）。ただし Kaggle のリーダーボードは時刻依存のため、`scraped_at` と `leaderboard_snapshot.csv` を run ごとに保存。

### NFR-3 セキュリティ
- `~/.kaggle/kaggle.json` は `security.md` の確認対象。読み取り専用、環境変数 fallback。
- ログに API key を出力しない。

### NFR-4 パフォーマンス
- 1 run で 500 episode 取得 ≤ 15 分（60 req/min × 500 ≈ 8.5 分 + ネットワーク余裕）。
- Parquet 書き込みは run 単位で 1 回（既存 recorder 方針と整合）。

### NFR-5 ストレージ
- 1 episode 平均 500 KB（gzip 後）想定、1,000 episode で 500 MB。`data/kaggle_episodes/` は `.gitignore` 対象。
- 1 run 1 parquet、hive パーティション `mode={1v1|ffa4}`。

### NFR-6 テスト
- HTTP 層はセッションを dependency injection してモック可能に。
- Dry-run 単独で E2E 的に通せる（replay 書き込みなし）。
- pytest 整備率 80%+（backend.md 方針）。

## Out of Scope

- **Meta Kaggle Dataset からの全履歴インポート** — 一次ルートで十分、必要時点で別計画。
- **行動ラベルの機械学習用前処理（parquet 化・特徴量抽出）** — 別プラン（模倣学習パイプライン）で扱う。
- **リプレイ可視化 UI** — 既存 `env.render("ipython")` を流用、本件で新規作成しない。
- **GitHub Actions / cron 自動化** — 本件は CLI ワンショット提供のみ。将来必要になれば Issue 化。
- **並列 DL** — レートリミット制約上、逐次で十分。

## Glossary

| Term | 説明 |
|---|---|
| Episode | Kaggle 上で実行された 1 試合の対戦記録。`episode_id` で一意 |
| EpisodeService | Kaggle 内部のリクエストサービス。`https://www.kaggle.com/requests/EpisodeService/*` |
| Leaderboard | 現在のコンペランキング。team 単位で表示、最新の best submission score を反映 |
| Submission ID | エージェント提出の一意 ID（Kaggle 内部） |
| Team ID | Kaggle 上のチーム ID（ソロ参加でも team_id が割り当てられる） |
| Rating (μ/σ) | Orbit Wars の TrueSkill 風レーティング。初期 μ₀=600 |
| Source | 本プロジェクト独自フィールド。selfplay / kaggle の区別 |
