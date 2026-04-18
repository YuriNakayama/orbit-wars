# Kaggle 提出自動化フロー — 要件定義

## 背景と目的

Orbit Wars コンペでは 1日最大5提出、かつ締切（2026-06-23 UTC 23:59）まで2ヶ月強。`pipeline/caseN/` で戦略を増やしていく計画のため、**caseディレクトリを指定するだけで検証 → tar.gz生成 → 認証 → Kaggle提出 → 結果確認まで一気通貫で走る自動化フロー** が必要。

目的:
1. 提出ミス（認証漏れ、ファイルパス間違い、エージェント壊れ）を仕組みで防ぐ。
2. CI 経由でも同じ手順で提出できるようにする（GitHub Actions から `gh workflow run`）。
3. 今回は `pipeline/case0`（Nearest Planet Sniper）を提出し、フロー全体が動作することを実証する。

## ユーザーストーリー

- **US-1**: 開発者として、`dev/submit case0 -m "first submission"` で `pipeline/case0` を検証し、Kaggle に提出したい。
- **US-2**: 開発者として、提出前にエージェントが最低1ターン例外なく動くことをローカルで確認したい。
- **US-3**: 開発者として、今日の提出数が5件に達していたら自動で停止してほしい。
- **US-4**: 開発者として、`--dry-run` で実際には提出せず、パッケージ内容と検証結果だけ確認したい。
- **US-5**: チームとして、`gh workflow run cd-kaggle-submit.yml -f case=case0 -f message="..."` で CI 経由でも提出できるようにしたい。
- **US-6**: 開発者として、提出後にvalidation状態をポーリングして成功/失敗を確認したい。

## 機能要件

### FR-1: case ディレクトリのバンドル
- 入力: `pipeline/<case>/`（`main.py` を含むディレクトリ）
- 出力: `data/submissions/<case>/<timestamp>.tar.gz`
- ルート直下に `main.py` を置く（Kaggle要求）。他の `.py` や `.md` も含める。`__pycache__/`、`*.log` は除外。
- 単一ファイルの場合は tar.gz ではなく `main.py` を直接提出する選択も可能（`--single-file`）。

### FR-2: ローカル検証（ドライラン）
- `kaggle_environments.make("orbit_wars")` で対戦1本走らせ、`env.run([<agent_path>, "random"])` が例外なく完了すること。
- `actTimeout` を超過しないこと（警告レベル、fail は任意）。
- 検証失敗時は提出を中止。

### FR-3: Kaggle 認証チェック
- `KAGGLE_USERNAME` + `KAGGLE_KEY` 環境変数、または `~/.kaggle/kaggle.json` の **いずれか** が存在することを確認。
- 両方なければ提出コマンドを実行せずに明示的エラー。

### FR-4: 提出前の5件/日チェック
- `kaggle competitions submissions orbit-wars -v` をパースし、**UTC基準で当日分の件数を数える**。
- 5件ある場合は `--force` がない限り中止。

### FR-5: 提出実行
- `kaggle competitions submit orbit-wars -f <file> -m "<message>"` を subprocess 実行。
- stdout/stderr を `rich` で表示、終了コード非ゼロならエラー。
- 提出成功時、返されるメッセージをパースして提出IDを取得（取れない場合は履歴から最新を推定）。

### FR-6: 提出後のポーリング（任意）
- `--wait` で有効化。最大N分（デフォルト5分）、30秒間隔で `kaggle competitions submissions` をポーリング。
- `status` が `pending` → `complete` / `error` になるまで待つ。

### FR-7: GitHub Actions ワークフロー
- トリガー: `workflow_dispatch`（手動）。
- 入力: `case` (文字列, 必須), `message` (文字列, 必須), `dry_run` (boolean, 既定 false), `wait` (boolean, 既定 true)。
- Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`。
- 実行内容: `uv sync` → `uv run python -m submit ...`。

### FR-8: `dev/submit` ラッパー
- 使い方: `dev/submit <case> -m "<msg>" [--dry-run] [--single-file] [--wait] [--force]`
- 内部で `uv run python -m submit submit ...` を呼ぶ。

## 非機能要件

- **Performance**: `dev/submit` 全体が60秒以内で完了（検証含む・ポーリング除く）。
- **Security**:
  - `KAGGLE_KEY` を絶対にログ出力しない。
  - `.env` / `~/.kaggle/kaggle.json` を**読み出して内容をechoしない**。
  - `security.md` ルール遵守。
- **Reliability**: Kaggle API 障害時は明確なエラー終了、ゴミ tar.gz を残さない。
- **Observability**: 全提出履歴を `data/submissions/<case>/submissions.jsonl` に追記（timestamp, case, message, result）。

## スコープ外

- caseN 共通の「複数バージョン管理」（今は timestamp 付きで保存するのみ）。
- ノートブック形式の提出（`kaggle competitions submit -k ...`）。必要になったら拡張。
- tar.gz の階層構造を自動検出する高度機能（`main.py` がルート前提）。
- `ci-backend.yml` の `working-directory: backend` 修正（既存バグ、別タスク）。

## 用語集

| 用語 | 意味 |
|------|------|
| case | `pipeline/caseN/` の単一戦略（main.py + 付随ファイル） |
| submission | Kaggle コンペへの1回分の提出 |
| dry-run | 提出せず検証とパッケージングだけ行うモード |
| overage time | 1エピソード中に共有される追加思考時間バジェット |
