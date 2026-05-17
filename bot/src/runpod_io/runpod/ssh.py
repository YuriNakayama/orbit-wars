"""RunPod pod の SSH endpoint 解決と ssh コマンド構築。

`runpod_sdk.get_pod()` が返す `runtime.ports` から TCP/22 の public host/port
を抽出し、ローカルから `ssh -i <key> root@<host> -p <port>` で接続できる
形に整える。`dev/runpod tail` の主な依存。

- pod が RUNNING で sshd 起動完了している必要がある (onstart.sh.tmpl 冒頭で
  start_sshd_early で立ち上げる)
- 既定の private key path は `~/.runpod/ssh/RunPod-Key-Go` (runpodctl が
  生成、アカウント側に登録済み)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SSH_KEY = Path.home() / ".runpod/ssh/RunPod-Key-Go"


class SshUnavailable(RuntimeError):
    """sshd 起動前 / pod 消失 / port 未公開 等で SSH できない場合に投げる。"""


@dataclass(frozen=True)
class SshEndpoint:
    """1 つの pod に対する SSH 接続情報。"""

    host: str
    port: int
    user: str = "root"
    key_path: Path = DEFAULT_SSH_KEY

    def to_command(self, remote_cmd: str | None = None) -> list[str]:
        """`ssh -i ... root@host -p port [remote_cmd]` の引数リストを返す。"""
        cmd: list[str] = [
            "ssh",
            "-i",
            str(self.key_path),
            "-p",
            str(self.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/tmp/runpod_known_hosts",
            "-o",
            "ConnectTimeout=10",
            f"{self.user}@{self.host}",
        ]
        if remote_cmd is not None:
            cmd.append(remote_cmd)
        return cmd


def parse_ssh_endpoint(pod: Mapping[str, Any]) -> SshEndpoint:
    """`runpod.get_pod()` 応答から SshEndpoint を抽出する。

    `runtime.ports` の中で privatePort=22 かつ isIpPublic=True のエントリを
    探す。見つからなければ SshUnavailable。
    """
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for entry in ports:
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("privatePort") or 0) != 22:
            continue
        if not entry.get("isIpPublic", False):
            continue
        host = str(entry.get("ip") or "")
        port = int(entry.get("publicPort") or 0)
        if host and port:
            return SshEndpoint(host=host, port=port)
    raise SshUnavailable(
        f"no public TCP/22 port for pod={pod.get('id')!r}; "
        "sshd may still be starting up or pod is not RUNNING"
    )


def get_pod_ssh_endpoint(sdk: Any, pod_id: str) -> SshEndpoint:
    """pod_id を SDK で引いて SshEndpoint に変換する。

    SDK は `runpod` モジュール (module-level state を持つ) または同等の
    `get_pod` 関数を持つもの。テスト時は MagicMock を渡せる。
    """
    pod = sdk.get_pod(pod_id)
    if pod is None:
        raise SshUnavailable(f"pod {pod_id!r} not found via SDK")
    if not isinstance(pod, Mapping):
        raise SshUnavailable(f"unexpected get_pod response type: {type(pod).__name__}")
    return parse_ssh_endpoint(pod)


def grab_onstart_log(
    endpoint: SshEndpoint,
    *,
    remote_path: str = "/var/log/onstart.log",
    timeout_seconds: float = 30.0,
    runner: Any = None,
) -> str:
    """SSH 経由で pod から `onstart.log` を取得する。

    watcher が stalled を検出したとき、`terminate_pod` を呼ぶ前にこの関数で
    ログを救出して S3 / ローカルに永久化する。`subprocess.run` を直接呼ぶと
    test しづらいので `runner` で差し替え可能 (defaults to subprocess.run)。

    `SshUnavailable` を投げる条件: SSH command が非ゼロ exit code を返した、
    または stdout が空。timeout や ssh エラーも同じ例外で包む。
    """
    import subprocess

    run = runner if runner is not None else subprocess.run
    cmd = endpoint.to_command(f"cat {remote_path}")
    try:
        result = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SshUnavailable(
            f"ssh cat {remote_path} timed out after {timeout_seconds}s"
        ) from exc
    except OSError as exc:
        raise SshUnavailable(f"ssh invocation failed: {exc}") from exc
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
        raise SshUnavailable(
            f"ssh cat {remote_path} exited {result.returncode}: "
            + " | ".join(stderr_tail)
        )
    if not result.stdout:
        raise SshUnavailable(f"ssh cat {remote_path} returned empty output")
    return result.stdout


__all__ = [
    "DEFAULT_SSH_KEY",
    "SshEndpoint",
    "SshUnavailable",
    "get_pod_ssh_endpoint",
    "grab_onstart_log",
    "parse_ssh_endpoint",
]
