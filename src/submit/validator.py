"""提出前にエージェントをローカルで軽く実行し、明らかな例外を検出する。"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, cast


class ValidationError(RuntimeError):
    """ローカル検証に失敗したときに投げる例外。"""


def _load_agent_module(main_py: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "orbit_wars_agent_under_test", main_py
    )
    if spec is None or spec.loader is None:
        raise ValidationError(f"モジュールロードに失敗: {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValidationError(
            f"main.py の import に失敗しました: {exc}\n{traceback.format_exc()}"
        ) from exc
    if not hasattr(module, "agent"):
        raise ValidationError("main.py に agent(obs) 関数が見つかりません。")
    return module


def dry_run(case_dir: Path, *, turns: int | None = None) -> dict[str, Any]:
    """エージェント vs random を短めに走らせる。

    Args:
        case_dir: pipeline/<case> ディレクトリ。
        turns: 実行ターン数。None なら既定（環境の episodeSteps）。

    Returns:
        実行結果の概要（最終ステップのステータスとタイミング）。
    Raises:
        ValidationError: 例外発生時。
    """

    main_py = (case_dir / "main.py").resolve()
    if not main_py.is_file():
        raise ValidationError(f"main.py が見つかりません: {main_py}")

    module = _load_agent_module(main_py)
    agent_fn = cast(Any, module.agent)

    try:
        from kaggle_environments import make
    except ImportError as exc:
        raise ValidationError(
            "kaggle_environments が import できません。`uv sync` を実行してください。"
        ) from exc

    configuration: dict[str, Any] = {}
    if turns is not None:
        configuration["episodeSteps"] = turns

    env = make("orbit_wars", configuration=configuration, debug=True)

    start = time.perf_counter()
    try:
        env.run([agent_fn, "random"])
    except Exception as exc:
        raise ValidationError(
            f"env.run で例外が発生しました: {exc}\n{traceback.format_exc()}"
        ) from exc
    elapsed = time.perf_counter() - start

    final = env.steps[-1]
    statuses = [s.status for s in final]
    return {
        "elapsed_s": round(elapsed, 3),
        "final_statuses": statuses,
        "steps": len(env.steps),
    }
