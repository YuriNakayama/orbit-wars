# リスクと依存

## リスク一覧

| # | リスク | 影響 | 確率 | 緩和策 |
|---|-------|------|------|-------|
| 1 | Kaggle認証未設定のまま submit が走り失敗 | 中 | 中 | `auth.ensure_credentials()` を最初に必ず呼ぶ。未設定なら明確エラー |
| 2 | 1日5提出の枠を浪費 | 高 | 中 | `kaggle_api.count_today()` を submit 前に呼ぶ。`--force` が無ければ停止 |
| 3 | tar.gz のルート階層がずれて Kaggle 側で `main.py` が見つからない | 高 | 低 | `packager` は常にルート直下に配置、pytestで検証 |
| 4 | エージェントがローカルで例外 → 提出後 validation 失敗 | 中 | 中 | `validator.dry_run` を必須化（`--skip-validation` は用意しない） |
| 5 | `KAGGLE_KEY` がログに漏れる | 高 | 低 | subprocess 呼び出しで env を明示せず継承、stdout のみ表示、stderr はパースのみ |
| 6 | CI でルールが未受諾のままsubmit | 中 | 低 | 事前に `kaggle competitions list --group entered` で確認 |
| 7 | Kaggle API 仕様変更で CLI 出力パース失敗 | 低 | 低 | `count_today()` はパース失敗時に conservative=5 扱い（安全側） |
| 8 | `data/submissions/` が肥大 | 低 | 高 | `.gitignore` で除外済（`data/` は既に ignore） |
| 9 | Kaggle ランタイム依存ライブラリが不足 | 中 | 低 | case0 は標準ライブラリのみ。将来は kaggle docker image を参照 |

## 外部依存

- **Kaggle API**: 障害時は提出不能 → 手動再実行
- **`kaggle` PyPI パッケージ**: 1.6 以降想定。バージョン固定は行わず最新を許容
- **`kaggle-environments`**: 既に依存。バージョン >=1.17.0

## 技術的負債

- `ci-backend.yml` の `working-directory: backend` は本プロジェクトに合っていない（既存の壊れた状態）。本タスクでは触らない。
- `dev/setup` も `backend/` 前提。同上。
- `src/submit/` はいずれ `src/` に他のエージェント実装が増えた時に名前空間衝突の恐れあり。その際 `src/kaggle_submit/` にリネーム検討。

## 未決事項

- 提出後の自動エピソード解析（`kaggle competitions episodes` → 勝率レポート）は将来タスク。
- 自己対戦で勝率しきい値を超えたら自動提出するフローも将来。

## 既知の挙動

- **tar.gz ERROR 問題 (2026-04-18, 解消済み)**: `case0` を README.md / agents.md 同梱の tar.gz で
  提出したところ `SubmissionStatus.ERROR` になった。対策として `packager.py` を
  ホワイトリスト方式（`*.py` / モデル重み / `*.json` / `*.yaml` / `*.txt` のみ）に改修し、
  さらに `tarfile.GNU_FORMAT` と決定的 tarinfo（uid/gid/mtime=0, mode=0o644）で生成するように変更。
  改修後に再提出した tar.gz は `SubmissionStatus.COMPLETE` (publicScore=600.0) を取得できた。
- **kaggle CLI 2.0.1 の submit 時の returncode**: 成功しても rc が非ゼロになるケースがあるため、
  `kaggle_api.submit()` は stdout に "Successfully submitted" を含むか、または stderr に
  100% アップロード完了マーカーがあれば成功扱いとする。加えて `confirm_submission()` で
  履歴 API の description 一致を確認するフェイルセーフを CLI フローに組み込んでいる。
