"""`uv run python -m submit` エントリポイント。"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from submit import (
    AuthError,
    ValidationError,
    build_archive,
    confirm_submission,
    dry_run,
    ensure_credentials,
    kaggle_submit,
    poll,
    record,
)
from submit.kaggle_api import KaggleCLIError

app = typer.Typer(
    help="Kaggle Orbit Wars への提出自動化CLI",
    no_args_is_help=True,
)
console = Console()

DEFAULT_PIPELINE_DIR = Path("pipeline")
DEFAULT_OUTPUT_DIR = Path("data/submissions")


def _resolve_case(case: str, *, pipeline_dir: Path) -> Path:
    case_dir = (pipeline_dir / case).resolve()
    if not case_dir.is_dir():
        console.print(f"[red]case ディレクトリが見つかりません:[/] {case_dir}")
        raise typer.Exit(code=2)
    return case_dir


@app.command("submit")
def submit_cmd(  # noqa: PLR0913 — CLI オプションは並列
    case: str = typer.Argument(
        ...,
        help=(
            "pipeline/<category>/<case> ディレクトリ名 "
            "(例: rulebase/case0, imitation/case1)"
        ),
    ),
    message: str = typer.Option(..., "-m", "--message", help="提出メッセージ"),
    dry_run_only: bool = typer.Option(
        False,
        "--dry-run",
        help="提出は行わず、検証とパッケージングのみ実行",
    ),
    single_file: bool = typer.Option(
        False,
        "--single-file",
        help="tar.gz ではなく main.py 単体で提出",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="提出後に validation 結果をポーリング",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="ローカルドライランをスキップ（非推奨）",
    ),
    pipeline_dir: Path = typer.Option(
        DEFAULT_PIPELINE_DIR,
        "--pipeline-dir",
        help="pipeline ルートディレクトリ",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        help="tar.gz 生成先",
    ),
) -> None:
    """指定した case を検証して Kaggle に提出する。"""

    case_dir = _resolve_case(case, pipeline_dir=pipeline_dir)
    console.print(f"[bold cyan]== {case} 提出フロー ==[/]")
    console.print(f"case dir : {case_dir}")
    console.print(f"dry-run  : {dry_run_only}")
    console.print(f"mode     : {'single-file' if single_file else 'tar.gz'}")

    if not skip_validation:
        console.print("\n[bold]1) ローカルドライラン[/]")
        try:
            dry = dry_run(case_dir)
        except ValidationError as exc:
            console.print(f"[red]検証失敗:[/]\n{exc}")
            raise typer.Exit(code=3) from exc
        console.print(
            f"  elapsed={dry['elapsed_s']}s steps={dry['steps']} "
            f"final_statuses={dry['final_statuses']}"
        )
    else:
        console.print("[yellow]検証をスキップしました[/]")

    console.print("\n[bold]2) アーカイブ生成[/]")
    archive = build_archive(
        case_dir,
        output_dir / case,
        single_file=single_file,
    )
    console.print(f"  生成: {archive}")

    if dry_run_only:
        record(
            base_dir=output_dir,
            case=case,
            message=message,
            archive=archive if not single_file else None,
            dry_run=True,
            result={"mode": "dry-run"},
        )
        console.print("[green]dry-run 完了。提出は行いませんでした。[/]")
        raise typer.Exit(code=0)

    console.print("\n[bold]3) 認証確認[/]")
    try:
        method = ensure_credentials()
    except AuthError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=4) from exc
    console.print(f"  認証方式: {method}")

    console.print("\n[bold]4) Kaggle へ提出[/]")
    try:
        stdout = kaggle_submit(archive, message)
    except KaggleCLIError as exc:
        console.print(f"[red]提出失敗:[/]\n{exc}")
        record(
            base_dir=output_dir,
            case=case,
            message=message,
            archive=archive if not single_file else None,
            dry_run=False,
            result={"error": str(exc)},
        )
        raise typer.Exit(code=7) from exc
    console.print(stdout)

    result: dict[str, object] = {"stdout": stdout}

    console.print("\n[bold]4b) 履歴APIで提出確認[/]")
    confirmed = confirm_submission(message)
    if confirmed is None:
        console.print(
            "[yellow]履歴に提出行が見つかりません。Kaggle UI を確認してください。[/]"
        )
        result["confirmed"] = False
    else:
        console.print(f"  確認: status={confirmed.get('status')}")
        result["confirmed"] = True
        result["confirmed_row"] = confirmed

    if wait:
        console.print("\n[bold]5) Validation ポーリング[/]")
        outcome = poll(message)
        result["poll"] = outcome
        console.print(f"  結果: {outcome.get('status')}")

    record(
        base_dir=output_dir,
        case=case,
        message=message,
        archive=archive if not single_file else None,
        dry_run=False,
        result=result,
    )
    console.print("\n[green]提出完了。[/]")


@app.command("submissions")
def submissions_cmd() -> None:
    """現在の提出一覧を取得する。"""

    from submit.kaggle_api import list_submissions

    try:
        rows = list_submissions()
    except KaggleCLIError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    for row in rows:
        console.print(row)


if __name__ == "__main__":  # pragma: no cover
    app()
