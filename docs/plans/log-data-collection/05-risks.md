# Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | Kaggle の `EpisodeService` が将来、認証を要求するようになる | High | Low | `kaggle_environments` 側の変更を追跡し、認証レイヤーを差し替え可能な `client.py` 構造を維持。CLI の `--session-cookie` 受け入れ hook を残しておく |
| 2 | レート制限超過で IP BAN（数時間〜数日） | High | Medium | 60 req/min をしっかり守る自前 token bucket、429 時は `urllib3.Retry` で自動 backoff、`--top` 既定値を保守的に（30）設定 |
| 3 | リプレイ JSON サイズが想定を大幅に超え、ストレージ破綻 | Medium | Low | gzip 圧縮で 5-10 倍縮、DL 前に `Content-Length` を見て閾値超過は skip、`--max-size-mb` オプション検討 |
| 4 | `EpisodeService` のレスポンス JSON フォーマット変更 | High | Medium | `records.py::build_match_record` で必須キー欠損時に `KaggleEpisodeError`、スキーマ検証テストを CI に組み込み、未知フィールドは無視 |
| 5 | リーダーボード CSV のカラム名が変わる | Medium | Low | `leaderboard.py` で `dict.get()` + 型変換で柔軟に対応、fixture を最新形式で常時更新 |
| 6 | 同一エピソードの重複取得（異 team 経由） | Low | High | `state.existing_episode_ids()` で事前 skip、in-memory set で run 内も防止。実装済みが MVP の要件 |
| 7 | コンペ終了（2026-06-23）後、履歴 API が段階的に閉じる可能性 | Medium | Medium | 可能な範囲で締め切り前に一次収集を完了させる。Meta Kaggle の `Episodes.csv` 経路を二次手段として準備する旨を README 明記 |
| 8 | Orbit Wars 特有の `mode` 判定（agents=2/4 以外）が将来追加される | Low | Low | `infer_mode` で未知 count は `ValueError`。対応時はマッピングを増やすだけ |
| 9 | Kaggle の利用規約違反疑義（スクレイピング扱い） | High | Low | 公式パッケージ `kaggle_environments.api` の関数を利用する範囲に留め、レート制御を厳守。`User-Agent` で目的を明記。TOS 確認を README に記載 |
| 10 | TrueSkill 風レーティングのフィールド名（μ/σ）が Orbit Wars で命名違う | Medium | Medium | Halite 踏襲の `updatedScore`/`updatedConfidence` を仮置きしつつ、生レスポンスを 1 件保存しておき後で確認 |
| 11 | `schema v2` 移行で selfplay の既存 Parquet と列数が不一致になり読めなくなる | High | Medium | 既存ファイルを `migrations/` に置き、`loader` で古スキーマも読めるよう `schema_version` 分岐。既存 selfplay ファイル破棄で済む場合は明示的に周知 |
| 12 | KeyboardInterrupt 時の flush が中途半端に書き込み、破損 Parquet 生成 | Medium | Low | `write_records` は一括書き込みなので、`try/finally` でバッファを必ず渡す。途中で 0 件なら flush しない |

## External Dependencies

- **Kaggle プラットフォーム**:
  - `kaggle_environments >= 1.17.0`（既存依存、API 変更を watch）
  - `kaggle` CLI（既存依存）
  - `https://www.kaggle.com/requests/EpisodeService/*` エンドポイントの可用性
  - `~/.kaggle/kaggle.json` の存在
- **OSS 参考実装**: `robga/simulations-episode-scraper`, `kuto0633/luxai2-episode-scraper`（API レスポンス形式の外部検証源として監視）
- **他 plan**: `baseline-reinforce` が模倣学習パイプラインで `data/kaggle_episodes/` を消費予定 → 本件の出力スキーマが決定する前に着手しないよう調整
- **コンペスケジュール**: 2026-06-23 最終提出締切。Scraping 完了は 2026-06-20 までを目標

## Technical Debt

- **HTTP クライアントの重複**: `kaggle_environments.api` にも類似実装があるが、セッション管理 + レート制御のために自前実装とする。将来公式が拡充すれば切り替え検討
- **Parquet schema v1/v2 併存**: 完全な下位互換を取るのではなく、v1 は v2 にマイグレーションするか破棄する方針で進めるため、マイグレーションスクリプト or ドキュメントが技術負債化し得る
- **レートリミッタの同期実装**: 将来 async 化する場合は `aiolimiter` 等へ置換が必要
- **`agent_names` が `kaggle_sub_{id}` のような合成値**: 模倣学習でラベル名として扱うときに submission_id 依存になる。別 plan で team_name ベースに拡張の余地

## Open Items

- **レート制限の正確な閾値**: 60 req/min は Halite 経験則。Orbit Wars 2026 時点の上限を実測で確認する必要あり（初回 run で監視）
- **EpisodeService レスポンス keys の最新形式**: Halite/Lux 時代と Orbit Wars 時代で差分がある可能性。初回で 1 件生 JSON を `data/kaggle_episodes/samples/` に保存して記録・分析
- **`rating_mu` / `rating_sigma` の厳密なフィールド名**: コンペの `abstract.md` は μ/σ をドキュメント、API 実体を要確認
- **最終提出後の継続収集判断**: 追加対戦期間（2026-06-24〜7-08）に新エピソードが生成されるため、その期間も収集を続けるかチームで合意要
- **`data/kaggle_episodes/` のサイズ上限運用ルール**: `dev/clean-data` 等の削除スクリプト必要性、`.gitignore` 管理方針
