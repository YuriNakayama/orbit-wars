# Log Data Collection — Web Technical Research

## 1. 公式 Kaggle API / Episode Service

### 1.1 Kaggle Public API（`kaggle` CLI）

公式ドキュメント ([Kaggle Public API](https://www.kaggle.com/docs/api), [kaggle-api GitHub](https://github.com/Kaggle/kaggle-api)) より、コンペ関連で利用する API:

| 機能 | CLI | 備考 |
|---|---|---|
| リーダーボード表示 | `kaggle competitions leaderboard -c orbit-wars --show` | 最新の全 submission rank / team name を CSV 取得 |
| 自身の提出履歴 | `kaggle competitions submissions -c orbit-wars -v` | CSV。既存 `list_submissions()` で利用 |
| 提出 | `kaggle competitions submit ...` | 既存 `submit/kaggle_api.py::submit` |

**認証**: `~/.kaggle/kaggle.json`（`username`, `key`）または `KAGGLE_USERNAME` / `KAGGLE_KEY` 環境変数。

**注**: リーダーボードには **team 単位** で行が出力される（`teamId`, `teamName`, `submissionDate`, `score`）。**submission_id** は CLI からは直接は取れないため、team_id → `list_episodes_for_team` で episode 一覧を引いて submission_id を逆算する経路が実用的。

### 1.2 kaggle_environments 内蔵 API（非公開だが公開エンドポイント）

`kaggle_environments/api.py`（バージョン 1.17.0+）に存在:

```python
import requests

BASE = "https://www.kaggle.com/requests/EpisodeService"

def get_episode_replay(episode_id: int) -> dict:
    return requests.post(f"{BASE}/GetEpisodeReplay", json={"EpisodeId": episode_id}).json()

def list_episodes(episode_ids: list[int]) -> dict:
    return requests.post(f"{BASE}/ListEpisodes", json={"Ids": episode_ids}).json()

def list_episodes_for_team(team_id: int) -> dict:
    return requests.post(f"{BASE}/ListEpisodes", json={"TeamId": team_id}).json()

def list_episodes_for_submission(submission_id: int) -> dict:
    return requests.post(f"{BASE}/ListEpisodes", json={"SubmissionId": submission_id}).json()
```

**レスポンスの想定構造**（Halite / Lux AI 参考実装より）:

- `ListEpisodes` レスポンス:
  ```json
  {
    "result": {
      "episodes": [
        {"id": 12345, "type": "OrbitWars", "createTime": "...", "endTime": "...",
         "agents": [{"id": 77, "submissionId": 888, "teamId": 99,
                     "initialScore": 600.0, "updatedScore": 625.4,
                     "updatedConfidence": 40.0, "state": "ACTIVE|TIMEOUT|ERROR"}]}
      ],
      "submissions": [{"id": 888, "teamId": 99, "isSubmissionOwner": false, ...}],
      "teams": [{"id": 99, "teamName": "...", ...}]
    }
  }
  ```
- `GetEpisodeReplay` レスポンス:
  ```json
  {"result": {"replay": "...JSON string with steps/configuration/info..."}}
  ```

`replay` フィールドは JSON 文字列。`json.loads(resp["result"]["replay"])` で kaggle_environments が期待する `{"steps": ..., "configuration": ..., "info": ...}` 形式が得られ、`loader.load_replay()` 相当の再構成が可能。

**認証**: **不要**。Web の kaggle.com 配下のリクエストと同じ扱い。ただし User-Agent を `Mozilla/...` にしておく方が 403 を回避しやすい（Halite スクレイパー実績）。

### 1.3 Meta Kaggle Dataset（代替路）

[Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle) は日次更新の公開データセット。simulation 関連テーブル:

| ファイル | サイズ | 内容 |
|---|---|---|
| `Episodes.csv` | 3.3 GB | 全 simulation episode の id/competition/type/createTime/endTime |
| `EpisodeAgents.csv` | 12 GB | 各 episode の agent 情報（submission_id, team_id, initial/updated rating, state） |
| `Competitions.csv` | — | competitionId → slug のマッピング（`orbit-wars` を抽出） |
| `Submissions.csv` | — | 提出メタデータ（publicScore など） |

`EpisodeAgents.csv` を `competitionId == orbit-wars` で絞り、`updatedScore` で sort、top N submission に対応する `EpisodeId` を `get_episode_replay` で取得、という経路が可能。

**Pros**: 全履歴取得可、スキルレーティング時系列を取れる。
**Cons**: 日次で完全リフレッシュ、最新 1 日分遅延、DL 15GB 級、CI で扱いづらい。

→ **本設計では二次活用とし、一次ルートは Episode Service API を採用**。将来、履歴全量が必要になった段階で `scripts/ingest_meta_kaggle.py` を別途追加する。

## 2. Similar OSS Projects

### 2.1 `robga/simulations-episode-scraper-match-downloader` (Kaggle Notebook)

- **Relevance**: Halite / ConnectX / Lux AI 等の Kaggle simulation 向けに広く使われるスクレイパー。本件と同じ `EpisodeService` を叩く。
- **Approach**:
  1. 対象コンペの `leaderboard` CSV から上位 N team を取得。
  2. 各 team に対し `list_episodes_for_team(team_id)` を呼び、episode_id と agent 情報を収集。
  3. 既取得 episode_id は skip、未取得のみ `get_episode_replay(ep_id)` で replay を DL。
  4. JSON を `data/raw/{episode_id}.json` として保存、メタデータを CSV に append。
- **Rate limit**: 60 req/min を遵守（`time.sleep(1)` + バーストカウンタ）。
- **Reusable patterns**:
  - team 単位スキャンで submission → episode を芋づる式に収集。
  - 既取得 ID セットを冒頭で読み込み、差分のみ処理する増分スキャン。
- **Pitfalls**:
  - User-Agent なしだと 403。
  - 同じ submission が複数 team との対戦で複数 episode を生むため、重複管理が必須。
  - TIMEOUT/ERROR 状態の episode は replay が欠損することがある。

### 2.2 `kuto0633/luxai2-episode-scraper-match-downloader` (Kaggle Notebook)

- **Relevance**: Lux AI 2 向け。Orbit Wars は Lux AI に近い 2v2/4-way のバトルロイヤル simulation で構造が類似。
- **Approach**: robga 版とほぼ同構成。追加で **watch list**（注目 team_id の手動指定）を持ち、学習用データを優先 DL。
- **Reusable patterns**:
  - `target_team_ids` の手動上書きオプション。
  - 進捗を tqdm + JSONL でログ。
- **Pitfalls**: 巨大リプレイ（数 MB）で OOM を避けるため DL 後即 gzip する実装。

### 2.3 `Kaggle/kaggle-environments` `api.py`

- **Relevance**: 本プロジェクトが依存する公式パッケージ。
- **Approach**: 上述の通り薄いラッパーのみ。リトライ・レート制御は持たない → 呼び出し側実装が必要。
- **Reusable patterns**: そのまま再利用。
- **Pitfalls**: 認証レイヤーもなく、失敗時は `requests` 例外がそのまま上がる。

### Pattern Comparison

| Aspect | Our Project | robga scraper | kaggle_environments | 採用方針 |
|---|---|---|---|---|
| HTTP 呼び出し | requests (新規) | requests | requests | `kaggle_environments.api` をそのまま呼ぶ |
| 認証 | `~/.kaggle/kaggle.json` (CLI) | なし | なし | CLI 側のみ認証 |
| レート制御 | 未実装 | sleep(1) + counter | なし | token bucket (60/60s) を新設 |
| レジューム | `data/matches/index.parquet` | CSV 読み直し | なし | Parquet スキャンで取得済 id 抽出 |
| 出力形式 | parquet + gzip JSON | JSON + CSV | dict | parquet + gzip JSON（既存踏襲） |

## 3. Library/Service Selection

### 3.1 HTTP + リトライ

| Candidate | Pros | Cons | Recommendation |
|---|---|---|---|
| ⭐ `requests` (既存・`kaggle_environments` 内で使用) | ゼロ追加、十分 | リトライ自前 | 採用。`urllib3.Retry` を `HTTPAdapter` でアタッチし 429/5xx に backoff |
| `httpx` | async 対応・豊富な機能 | 依存追加 | 将来 async 化時に検討 |
| `tenacity` | デコレータ型リトライ | 追加依存 | 既存コードが少ない方針なので見送り |

### 3.2 レートリミット

| Candidate | Pros | Cons | Recommendation |
|---|---|---|---|
| 自前 token bucket (`time.monotonic`) | 依存なし・40 行程度 | ロジック管理自前 | ⭐ 採用 |
| `ratelimit` PyPI | 宣言的デコレータ | 未メンテ傾向、細かい制御弱 | 不採用 |
| `aiolimiter` | async 向け | 同期要件と不整合 | 不採用 |

### 3.3 リーダーボード取得

| Candidate | Pros | Cons | Recommendation |
|---|---|---|---|
| ⭐ `kaggle competitions leaderboard --show` subprocess | 既存 `submit/kaggle_api.py` と整合、認証統一 | 解析が文字列 | 採用。CSV 形式で stdout を parse |
| Kaggle Public API の REST 直叩き | subprocess 不要 | 実装重 | 不採用 |
| `kaggle.api` Python SDK | 直接 Python から呼べる | 既存方針（subprocess wrap）と非一貫 | 不採用 |

## 4. API/Protocol 詳細

### 4.1 `EpisodeService` 呼び出しの安定化

Halite/Lux 参考実装から得られたプラクティス:

```python
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "orbit-wars-log-collector/0.1",
    "Content-Type": "application/json",
})
adapter = HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
))
SESSION.mount("https://", adapter)
```

これを `src/env/kaggle/client.py` にセッション singleton として保持、`kaggle_environments.api` のモジュール変数 `requests` を monkeypatch せず、**薄く再実装したクライアント関数**を用意する（`kaggle_environments.api` に依存しないことで test しやすくなる）。

### 4.2 レスポンス解析

`result.episodes[i].agents[j].state` は以下のいずれか（Halite 観測値より推定）:
- `"ACTIVE"` / `"INACTIVE"` / `"TIMEOUT"` / `"ERROR"` / `"INVALID"`

`replay` 文字列を `json.loads()` した後、以下を期待:
```json
{
  "name": "orbit_wars",
  "configuration": {"agents": 2, "seed": ...},
  "steps": [[{"action": [...], "observation": {...}, "status": "ACTIVE", "reward": ...}, ...], ...],
  "rewards": [...],
  "statuses": [...],
  "info": {...}
}
```

この形式は `kaggle_environments.make("orbit_wars", steps=steps, configuration=configuration)` で直接 env を再構成可能。既存 `loader.load_replay()` と互換。

## 5. Research Summary

- **一次ルート**: `kaggle_environments.api` の `list_episodes_for_team` + `get_episode_replay` を使用。認証不要、実装が軽い。
- **リーダーボード取得**: `kaggle competitions leaderboard -c orbit-wars --show` の CSV stdout を parse。
- **レート制御**: 自前 token bucket で 60 req/60s を保守、429 は `urllib3.Retry` で backoff。
- **レジューム**: `data/kaggle_episodes/index.parquet` から `episode_id` unique を引いて skip set にする。
- **メタデータ**: `MatchRecord` を拡張（`episode_id`, `submission_id`, `team_id`, `rating_mu`, `rating_sigma`, `source`, `scraped_at`）。`SCHEMA_VERSION` を 2 に上げる。
- **Meta Kaggle 経路は二次扱い**。将来、時系列レーティング解析や全履歴学習が必要になった時点で追加。
- **推奨アーキテクチャ**: 新モジュール `src/env/kaggle/` に `client.py`（HTTP）、`leaderboard.py`（CLI ラッパー）、`scraper.py`（統合ロジック）、`records.py`（MatchRecord 変換）、`cli.py`（typer）を配置。合計 400-500 行。既存 `env/recorder.py` の関数を再利用。

## Sources

- [Kaggle Public API](https://www.kaggle.com/docs/api)
- [Kaggle/kaggle-api GitHub](https://github.com/Kaggle/kaggle-api)
- [Kaggle/kaggle-environments GitHub](https://github.com/Kaggle/kaggle-environments)
- [kaggle_environments api.py (EpisodeService ラッパー)](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/api.py)
- [robga/simulations-episode-scraper-match-downloader (Kaggle Notebook)](https://www.kaggle.com/code/robga/simulations-episode-scraper-match-downloader)
- [kuto0633/luxai2-episode-scraper-match-downloader (Kaggle Notebook)](https://www.kaggle.com/code/kuto0633/luxai2-episode-scraper-match-downloader)
- [Meta Kaggle Dataset](https://www.kaggle.com/datasets/kaggle/meta-kaggle)
- [DeepWiki: Kaggle Competitions API](https://deepwiki.com/Kaggle/kaggle-api/3.1-competitions-api)
