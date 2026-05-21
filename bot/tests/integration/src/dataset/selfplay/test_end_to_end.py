"""Self-play CLI → recorder → loader → replay reconstruction smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataset.cli import app
from dataset.storage import loader, recorder


class _FakeFile:
    def __init__(self, key: str, mode: str, backing: dict[str, bytes]) -> None:
        self._key = key
        self._mode = mode
        self._backing = backing
        self._buf = bytearray()

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_exc: object) -> None:
        if "w" in self._mode:
            self._backing[self._key] = bytes(self._buf)

    def write(self, data: bytes) -> int:
        self._buf.extend(data)
        return len(data)

    def read(self) -> bytes:
        return self._backing[self._key]


class _FakeFS:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def open(self, path: str, mode: str = "rb"):  # type: ignore[no-untyped-def]
        return _FakeFile(path.removeprefix("s3://"), mode, self._store)


@pytest.mark.slow
def test_run_then_list_then_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: dict[str, bytes] = {}
    monkeypatch.setattr(recorder, "_s3", lambda: _FakeFS(store))
    monkeypatch.setattr(loader, "_s3", lambda: _FakeFS(store))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--agents",
            "random,random",
            "--mode",
            "1v1",
            "-n",
            "2",
            "--parallel",
            "2",
            "--save-replay",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    df = loader.list_matches(tmp_path, mode="1v1")
    assert df.height == 2

    match_id = df["match_id"][0]
    env = loader.load_replay(match_id, source="selfplay")
    rendered = env.render(mode="json")
    assert isinstance(rendered, (dict, str))
