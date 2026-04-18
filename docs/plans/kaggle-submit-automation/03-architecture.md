# Kaggle 提出自動化フロー — アーキテクチャ設計

## 全体図

```
┌──────────────┐    ┌───────────────────────────────────────────┐
│ dev/submit   │───▶│ uv run python -m submit submit …          │
└──────────────┘    │                                             │
      ▲              │  src/submit/__main__.py  (typer CLI)       │
      │              │    submit()  ─┬─▶ packager.build_archive() │
      │              │               ├─▶ validator.dry_run()      │
      │              │               ├─▶ auth.ensure_credentials()│
      │              │               ├─▶ kaggle_api.count_today() │
      │              │               ├─▶ kaggle_api.submit()      │
      │              │               └─▶ kaggle_api.poll()        │
      │              └───────────────────────────────────────────┘
      │
┌──────────────┐      ┌─────────────────────────────────────────────┐
│ GitHub UI    │─────▶│ .github/workflows/cd-kaggle-submit.yml       │
│  workflow    │      │  (workflow_dispatch) → uv sync → submit      │
│  dispatch    │      │  Secrets: KAGGLE_USERNAME, KAGGLE_KEY        │
└──────────────┘      └─────────────────────────────────────────────┘
```

## ディレクトリ構造

```
src/
  submit/
    __init__.py
    __main__.py        # typer app (python -m submit)
    auth.py            # 認証有無確認
    kaggle_api.py      # kaggle CLI ラッパー (subprocess)
    packager.py        # tar.gz 生成 / 単一ファイル選択
    validator.py       # kaggle_environments でドライラン
    history.py         # submissions.jsonl 追記
dev/
  submit               # bash ラッパー
.github/
  workflows/
    cd-kaggle-submit.yml
tests/
  submit/
    test_packager.py
    test_auth.py
    test_validator.py
    test_history.py
    __init__.py
data/
  submissions/         # gitignore（大きい）
    <case>/
      <YYYYMMDD-HHMMSS>.tar.gz
      submissions.jsonl
```

## モジュール設計

### `src/submit/packager.py`

```python
from pathlib import Path
import tarfile

EXCLUDE_PATTERNS = {"__pycache__", "*.pyc", "*.log", ".DS_Store"}

def build_archive(case_dir: Path, out_dir: Path, single_file: bool = False) -> Path:
    """pipeline/<case>/ を tar.gz に固める。main.py がルート直下に来る。

    Returns: 生成ファイルのパス (tar.gz または main.py)
    Raises: FileNotFoundError, ValueError
    """
```

- `main.py` がルートディレクトリに存在することを確認。
- `single_file=True` ならファイルをそのまま返す。
- `single_file=False` なら `<case>_<timestamp>.tar.gz` を作り、`case_dir` 内のファイル（除外パターン以外）を **ルート直下に** 配置。

### `src/submit/validator.py`

```python
def dry_run(case_dir: Path, turns: int = 10) -> None:
    """kaggle_environments.make で agent vs random を実行し、例外がないことを確認。
    Raises: ValidationError（例外メッセージとスタックトレース付き）。"""
```

- `main.py` を `importlib` でロードし、`agent` 関数を取得。
- `env.run([agent_fn, "random"])` を実行、`env.steps[-1]` の `status` が全員 `DONE` または `ACTIVE` であることを確認。
- タイムアウト（`actTimeout=1`）は実時間で計測し、超えた場合は警告のみ。

### `src/submit/auth.py`

```python
def ensure_credentials() -> None:
    """Kaggle 認証が設定されているかをチェックする。
    優先順位: 環境変数 KAGGLE_USERNAME/KAGGLE_KEY → ~/.kaggle/kaggle.json。
    未設定なら AuthError を raise（内容はログに出さない）。"""
```

### `src/submit/kaggle_api.py`

```python
COMPETITION = "orbit-wars"

def submit(file_path: Path, message: str) -> str: ...
def list_submissions() -> list[dict]: ...
def count_today() -> int: ...
def poll(submission_id: str, timeout_s: int = 300, interval_s: int = 30) -> dict: ...
```

- 内部で `subprocess.run(["kaggle", "competitions", ...], check=True, capture_output=True)`。
- 失敗時は stderr をそのままエラーに載せる（認証情報は含まれない前提）。
- `list_submissions` は `--csv` 風の出力をパース（`-v` フラグで CSV になる）。

### `src/submit/history.py`

```python
def record(case: str, message: str, archive: Path, result: dict) -> None:
    """data/submissions/<case>/submissions.jsonl に1行追記。"""
```

### `src/submit/__main__.py`

```python
import typer
from rich.console import Console

app = typer.Typer()

@app.command()
def submit(
    case: str = typer.Argument(..., help="pipeline/<case> ディレクトリ名"),
    message: str = typer.Option(..., "-m", "--message"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    single_file: bool = typer.Option(False, "--single-file"),
    wait: bool = typer.Option(False, "--wait"),
    force: bool = typer.Option(False, "--force", help="5件/日チェックをスキップ"),
) -> None:
    ...
```

## 外部連携

- **Kaggle API**: CLI 経由。認証は env 変数優先。
- **GitHub Actions**: `workflow_dispatch` のみ。`KAGGLE_USERNAME`, `KAGGLE_KEY` を secrets で渡す。
- **kaggle-environments**: ローカル検証に使用（依存は既存）。

## データモデル

### `submissions.jsonl` 1行の schema

```json
{
  "timestamp": "2026-04-18T16:40:00Z",
  "case": "case0",
  "message": "Nearest planet sniper v1",
  "archive": "data/submissions/case0/20260418-164000.tar.gz",
  "dry_run": false,
  "submission_id": "12345678",
  "status": "complete"
}
```

## 依存追加

- `pyproject.toml` の `dependencies` に `kaggle>=1.6` を追加。
- `uv add kaggle` で sync する（本タスク内で実施）。

## セキュリティ

- `.env`, `~/.kaggle/kaggle.json` は読み出してはならない（security.md）。存在チェックと `os.environ` 参照のみで判定。
- ログ出力で `KAGGLE_KEY` を絶対に含めない。`subprocess` 呼び出しで `env` を継承するだけ。
- `~/.kaggle/kaggle.json` が存在するかは `Path.exists()` で確認するのみ。中身を開かない。
