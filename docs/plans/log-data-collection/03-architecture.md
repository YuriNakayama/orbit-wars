# Log Data Collection — Architecture Design

既存 `src/env/` の評価基盤と責務分離しつつ、既存の永続化/ロード系と整合した **Kaggle リプレイ取得モジュール** を `src/env/kaggle/` として追加する。`MatchRecord` / `recorder` を拡張し、新ソース (`source="kaggle"`) の行が selfplay と共存できるようにする。

## 1. 全体構成

```
                ┌──────────────────────────────────────────────┐
                │   CLI (src/env/kaggle/cli.py)                │
                │   python -m env.kaggle scrape|list|inspect   │
                └───────────────────┬──────────────────────────┘
                                    │
        ┌───────────────────────────┼─────────────────────────┐
        ▼                           ▼                         ▼
  ┌───────────────┐           ┌───────────────┐         ┌───────────────┐
  │ leaderboard   │           │   scraper     │         │   records     │
  │ (kaggle CLI)  │──team_ids►│ (orchestrate) │──rows──►│ (MatchRecord) │
  └───────────────┘           └───────┬───────┘         └───────┬───────┘
                                      │                         │
                          ┌───────────┼──────────────┐          │
                          ▼           ▼              ▼          │
                    ┌──────────┐ ┌──────────┐ ┌───────────┐     │
                    │  client  │ │rate_limit│ │   state   │     │
                    │ (HTTP)   │ │(bucket)  │ │(resume set)│    │
                    └────┬─────┘ └──────────┘ └─────┬─────┘     │
                         │                          │           │
                         ▼                          ▼           ▼
                 https://www.kaggle.com    data/kaggle_    src/env/recorder.py
                 /requests/EpisodeService  episodes/       (shared)
                         │                  index.parquet
                         ▼                          │
                 gzip(replay JSON) ────────────────►┤
                                                    ▼
                                   data/kaggle_episodes/matches/
                                   ├── index.parquet/mode={1v1|ffa4}/run_*.parquet
                                   └── replays/{episode_id}.json.gz
```

## 2. 責務分割

| モジュール | 責務 | 想定行数 |
|---|---|---|
| `src/env/kaggle/__init__.py` | 公開 API (`scrape`, `load_episode`) | 〜30 |
| `src/env/kaggle/client.py` | `requests.Session` + EpisodeService POST ラッパー。`get_episode_replay`, `list_episodes_for_team` | 〜120 |
| `src/env/kaggle/rate_limit.py` | 自前 token bucket (60/60s)。`with bucket.acquire(): ...` | 〜60 |
| `src/env/kaggle/leaderboard.py` | `kaggle competitions leaderboard --show` subprocess ラッパー。CSV → `list[TeamRank]` | 〜80 |
| `src/env/kaggle/state.py` | 既取得 episode_id 抽出。`pl.scan_parquet` で unique id set を返す | 〜60 |
| `src/env/kaggle/records.py` | Episode API レスポンス → 拡張 `MatchRecord`。mode 判定・rating 抽出 | 〜150 |
| `src/env/kaggle/scraper.py` | orchestration: leaderboard → team_ids → episodes → replays → records | 〜180 |
| `src/env/kaggle/cli.py` | typer CLI (`scrape`, `list`, `inspect`) | 〜120 |
| `src/env/kaggle/types.py` | `TeamRank`, `EpisodeMeta`, `ScrapeSpec` frozen dataclass | 〜80 |

**合計** 〜880 行。1 ファイル 200 行前後を維持。

既存資産との統合:
- `src/env/types.py::MatchRecord` を拡張（`SCHEMA_VERSION=2`）。
- `src/env/recorder.py::write_records` / `write_replay` を汎用化（`data_root` 下の相対パスを引数化して `matches/` 以外にも使えるように）。
- `src/env/loader.py` に `load_kaggle_replay(episode_id)` を追加（`data/kaggle_episodes/replays/{episode_id}.json.gz` を読む）。

## 3. データフロー

### 3.1 Scrape フロー（主経路）

1. `cli.py::scrape` が `--top`, `--modes`, `--limit-per-team`, `--dry-run`, `--data-root` を typer でパース。
2. `leaderboard.fetch(top=N)` が `kaggle competitions leaderboard -c orbit-wars --show` を subprocess 実行、CSV stdout を parse して `list[TeamRank]` を返す。同時に `data/kaggle_episodes/leaderboards/{run_id}.csv` に生の CSV を保存。
3. `state.existing_episode_ids(data_root, modes)` が Parquet index を scan して `set[int]` を返す（既取得 id）。
4. `scraper.collect(spec, teams, existing_ids)` が team 単位で:
   - `client.list_episodes_for_team(team_id)` を call（`rate_limit.acquire()` 経由）。
   - レスポンスから `EpisodeMeta` を構築、modes フィルタ・既取得除外を適用。
   - 未取得の各 episode について `client.get_episode_replay(episode_id)` を call。
   - replay JSON 文字列を gzip 圧縮し `replay_bytes[episode_id] = gzipped` に貯める。
   - メタを `records.build_match_record(episode_meta, team_info)` で `MatchRecord` に変換。
5. `--dry-run` でなければ `recorder.write_run(records, replay_bytes, data_root_with_prefix)` で Parquet + gzip を永続化。
6. rich.Table で summary (新規取得数、mode 内訳、総サイズ) を stdout に表示。

### 3.2 CLI 実行例

```
uv run python -m env.kaggle scrape --top 30 --modes 1v1,ffa4
uv run python -m env.kaggle scrape --top 10 --dry-run
uv run python -m env.kaggle list --mode 1v1 --limit 20
uv run python -m env.kaggle inspect 12345
```

### 3.3 Resume フロー

- 既取得 id を `pl.scan_parquet(index_root / "**/*.parquet").select("episode_id").unique().to_series().to_list()` で取得。
- `scraper` 内で `if ep.id in existing_ids: continue`。
- KeyboardInterrupt はキャッチし、収集済みバッファを最終 run_id に書き込んで正常終了。

### 3.4 エラーパス

- `client.post(endpoint, body)` は `requests.exceptions.RequestException` を包んで `KaggleEpisodeError` として raise。
- `scraper` レベルで 1 episode の失敗は `logger.warning(...)` にして継続。
- 連続 5 件失敗で `scraper` 自体を終了。
- 429 は `urllib3.Retry` が自動 backoff。

## 4. Frontend 設計

UI は持たない。既存方針（ノートブック + `env.render`）を踏襲。

- `pipeline/case1/eda/kaggle_replay_viewer.py`（percent format）を新設:
  ```python
  # %%
  from env import loader
  from env.kaggle import load_episode
  # %%
  df = loader.list_matches(data_root=Path("data/kaggle_episodes"), mode="1v1", limit=20)
  df.head(20)
  # %%
  episode_id = int(df["episode_id"][0])
  env = load_episode(episode_id, data_root=Path("data/kaggle_episodes"))
  env.render(mode="ipython", width=800, height=600)
  ```

## 5. Backend 設計

### 5.1 `client.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

BASE_URL = "https://www.kaggle.com/requests/EpisodeService"


class KaggleEpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClientConfig:
    user_agent: str = "orbit-wars-log-collector/0.1"
    timeout_sec: float = 30.0
    max_retries: int = 5


def build_session(config: ClientConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.user_agent,
        "Content-Type": "application/json",
    })
    retry = Retry(
        total=config.max_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _post(session: requests.Session, path: str, body: dict[str, Any],
          timeout: float) -> dict[str, Any]:
    url = f"{BASE_URL}/{path}"
    try:
        resp = session.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise KaggleEpisodeError(f"{path} failed: {exc}") from exc
    return resp.json()


def list_episodes_for_team(session: requests.Session, team_id: int,
                           *, timeout: float = 30.0) -> dict[str, Any]:
    return _post(session, "ListEpisodes", {"TeamId": team_id}, timeout)


def get_episode_replay(session: requests.Session, episode_id: int,
                       *, timeout: float = 30.0) -> dict[str, Any]:
    return _post(session, "GetEpisodeReplay", {"EpisodeId": episode_id}, timeout)


def extract_replay_json(response: dict[str, Any]) -> str:
    """GetEpisodeReplay のレスポンスから replay JSON 文字列を取り出す。"""
    replay = response.get("result", {}).get("replay")
    if not isinstance(replay, str):
        raise KaggleEpisodeError("replay field missing or not a string")
    return replay
```

### 5.2 `rate_limit.py`

```python
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TokenBucket:
    """60 req / 60 s のような粗い制御向け。スレッドセーフ。"""
    capacity: int
    window_sec: float

    def __post_init__(self) -> None:
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if t > now - self.window_sec]
                if len(self._timestamps) < self.capacity:
                    self._timestamps.append(now)
                    break
                wait = self.window_sec - (now - self._timestamps[0])
            time.sleep(max(wait, 0.05))
        yield
```

### 5.3 `leaderboard.py`

```python
from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path


COMPETITION = "orbit-wars"


@dataclass(frozen=True)
class TeamRank:
    rank: int
    team_id: int
    team_name: str
    score: float
    submission_date: str


def fetch(top: int, *, raw_snapshot: Path | None = None) -> list[TeamRank]:
    proc = subprocess.run(  # noqa: S603
        ["kaggle", "competitions", "leaderboard", "-c", COMPETITION, "--show", "--csv"],
        capture_output=True, text=True, check=True,
    )
    if raw_snapshot is not None:
        raw_snapshot.parent.mkdir(parents=True, exist_ok=True)
        raw_snapshot.write_text(proc.stdout, encoding="utf-8")
    reader = csv.DictReader(io.StringIO(proc.stdout))
    ranks: list[TeamRank] = []
    for i, row in enumerate(reader, start=1):
        if i > top:
            break
        ranks.append(TeamRank(
            rank=i,
            team_id=int(row["teamId"]),
            team_name=row.get("teamName", ""),
            score=float(row.get("score", 0.0)),
            submission_date=row.get("submissionDate", ""),
        ))
    return ranks
```

### 5.4 `records.py`

```python
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from env.types import MatchRecord, AgentTiming, MAX_PLAYERS


MODE_BY_AGENT_COUNT = {2: "1v1", 4: "ffa4"}


def infer_mode(agent_count: int) -> str:
    if agent_count not in MODE_BY_AGENT_COUNT:
        raise ValueError(f"unsupported agent count: {agent_count}")
    return MODE_BY_AGENT_COUNT[agent_count]


def build_match_record(
    episode: dict[str, Any],
    *,
    run_id: str,
    scraped_at: str,
) -> MatchRecord:
    agents = episode.get("agents", [])
    mode = infer_mode(len(agents))
    winner, scores = _resolve_outcome(agents)
    timings = tuple(AgentTiming(timeouts=0, p50=0.0, p95=0.0, max=0.0) for _ in agents)

    episode_id = int(episode["id"])
    started_at = episode.get("createTime") or scraped_at
    elapsed_sec = _elapsed(episode.get("createTime"), episode.get("endTime"))

    return MatchRecord(
        match_id=f"kaggle_ep_{episode_id}",
        run_id=run_id,
        mode=mode,
        seed=-1,
        started_at=started_at,
        elapsed_sec=elapsed_sec,
        turns=int(episode.get("turns") or 0),
        winner=winner,
        draw=(winner < 0),
        agent_names=tuple(_agent_label(a) for a in agents),
        agent_versions=tuple(str(a.get("submissionId") or "") for a in agents),
        agent_scores=tuple(int(a.get("finalScore") or 0) for a in agents),
        agent_timings=timings,
        replay_path=f"replays/{episode_id}.json.gz",
        git_sha="",
        # v2 拡張
        source="kaggle",
        episode_id=episode_id,
        scraped_at=scraped_at,
        agent_submission_ids=tuple(int(a.get("submissionId") or 0) for a in agents),
        agent_team_ids=tuple(int(a.get("teamId") or 0) for a in agents),
        agent_rating_mus=tuple(float(a.get("updatedScore") or 0.0) for a in agents),
        agent_rating_sigmas=tuple(float(a.get("updatedConfidence") or 0.0) for a in agents),
        agent_states=tuple(str(a.get("state") or "") for a in agents),
    )


def _agent_label(agent: dict[str, Any]) -> str:
    sub = agent.get("submissionId")
    return f"kaggle_sub_{sub}" if sub is not None else "kaggle_unknown"


def _resolve_outcome(agents: list[dict[str, Any]]) -> tuple[int, tuple[int, ...]]:
    ...  # state == "ACTIVE" の数と finalScore から winner 決定
```

`MatchRecord` の v2 拡張は `src/env/types.py` 側で `agent_{i}_submission_id`, `agent_{i}_team_id`, `agent_{i}_rating_mu`, `agent_{i}_rating_sigma`, `agent_{i}_state` を `to_row()` に追加。`source`, `episode_id`, `scraped_at` はトップレベル列に追加。

### 5.5 `scraper.py`

```python
from __future__ import annotations

import datetime as dt
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress

from env.kaggle import client, leaderboard, rate_limit, records, state
from env.types import MatchRecord


@dataclass(frozen=True)
class ScrapeSpec:
    top: int
    modes: tuple[str, ...]
    limit_per_team: int | None
    data_root: Path
    dry_run: bool
    include_failed: bool


@dataclass(frozen=True)
class ScrapeResult:
    run_id: str
    records: tuple[MatchRecord, ...]
    dry_run_preview: dict[str, int]


def run(spec: ScrapeSpec, progress: Progress | None = None) -> ScrapeResult:
    ...  # leaderboard → per-team loop → per-episode loop
```

### 5.6 `types.py`

```python
@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: int
    mode: str
    create_time: str
    end_time: str
    agents: tuple[dict[str, Any], ...]
    # state == "ERROR" 等のフィルタ用
```

## 6. データモデル

### 6.1 ストレージレイアウト

```
data/kaggle_episodes/
├── leaderboards/
│   └── {run_id}.csv                    # leaderboard スナップショット
└── matches/
    ├── index.parquet/
    │   └── mode={1v1|ffa4}/
    │       └── run_{run_id}[_N].parquet   # MatchRecord v2 rows
    └── replays/
        └── {episode_id}.json.gz           # replay JSON (gzip)
```

`data/matches/` と同構造にすることで、既存 `loader.list_matches(data_root=Path("data/kaggle_episodes"))` がそのまま動作する。

### 6.2 スキーマ v2 追加列

トップレベル:
- `schema_version`: 2
- `source`: `"selfplay" | "kaggle"`
- `episode_id`: int（selfplay は -1）
- `scraped_at`: str（ISO 8601, selfplay は `started_at` と同値）

各 agent (i = 0..3):
- `agent_{i}_submission_id`: int
- `agent_{i}_team_id`: int
- `agent_{i}_rating_mu`: float
- `agent_{i}_rating_sigma`: float
- `agent_{i}_state`: str

### 6.3 Replay ペイロード

`data/kaggle_episodes/replays/{episode_id}.json.gz` は `json.loads(result.replay)` した結果を `json.dumps` し直して gzip。これにより `loader.load_replay_payload()` が:
- `payload["name"] = "orbit_wars"`
- `payload["configuration"]`
- `payload["steps"]`

を期待する形式と一致し、`loader.load_replay()` で env 再構成可能。

## 7. インフラ / 外部統合

- **AWS リソース**: 無し。ローカル FS のみ。
- **外部サービス**:
  - `https://www.kaggle.com/requests/EpisodeService/*`（認証不要、POST）
  - `kaggle competitions leaderboard` CLI（`~/.kaggle/kaggle.json` 必須）
- **依存追加**: 無し。`requests` は kaggle_environments の推移的依存で既に存在。
- **CI**: `dev/test-backend` に新モジュール pytest を組み込み。実 API は叩かない（responses / requests-mock）。
- **ルール**: `.claude/rules/backend.md`（strict mypy, Python 3.13, 無例外で `Any` 禁止）、`.claude/rules/security.md`（`.kaggle/kaggle.json` 保護）。

## 8. 既存プランとの整合

- **evaluation-system**: `src/env/` の共通基盤を使う。本件はその下位モジュール `src/env/kaggle/`。
- **baseline-reinforce**: 模倣学習の教師データとして、本件が出力する `data/kaggle_episodes/` を消費予定。スキーマを `MatchRecord` 互換にすることで既存 loader を流用。
- **kaggle-submit-automation**: 既存の `src/submit/` 群とは役割が別（提出 vs ログ取得）。CLI も分離。
