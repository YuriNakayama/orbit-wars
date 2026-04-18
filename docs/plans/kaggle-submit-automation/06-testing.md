# テスト戦略

## アプローチ

- ユニットテスト主体。Kaggle API 呼び出しは全てモック（`unittest.mock.patch`）。
- ドライラン（`validator.dry_run`）は `case0/main.py` を実際にロードしてスモークテスト扱い。
- CI では `KAGGLE_USERNAME/KAGGLE_KEY` が未設定でもテストが通るよう、`auth` テストは env を monkeypatch。

## ユニットテスト

### Backend (`tests/submit/`)

| テストファイル | 対象 |
|---------------|------|
| `test_packager.py` | tar.gz 生成・除外パターン・単一ファイルモード |
| `test_auth.py` | 環境変数ありパス／config file ありパス／両方なしで AuthError |
| `test_validator.py` | case0 を対象にスモークテスト（1エピソード完走） |
| `test_history.py` | jsonl 追記、ディレクトリ自動作成 |
| `test_kaggle_api.py` | subprocess を monkeypatch、count_today のパース |

## 統合テスト

- 本タスクでは手動。 `dev/submit case0 --dry-run` で確認。
- Secrets 設定後に `gh workflow run cd-kaggle-submit.yml -f case=case0 -f dry_run=true`。

## E2E

- 実提出が E2E に相当。本リリースで1回だけ（case0）実施。

## テストデータ

- `pipeline/case0/main.py` を fixture 代わりに使う。

## カバレッジ目標

- `src/submit/` ユニット: 80% 以上
- 全体: 強制しない（実装まだ無し）
