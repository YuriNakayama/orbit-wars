# runpod-basis — Web Technical Research

## Official Documentation

### RunPod Python SDK (`runpod`)

- **PyPI**: [`runpod`](https://pypi.org/project/runpod/) — pip / uv add 可能。GitHub: [runpod/runpod-python](https://github.com/runpod/runpod-python)。
- **核心 API** ([Manage Pods - Runpod Docs](https://docs.runpod.io/pods/manage-pods)):
  - `runpod.api_key = "..."` で API key 設定 (環境変数 `RUNPOD_API_KEY` も読まれる)。
  - `runpod.create_pod(name, image_name, gpu_type_id, ...) -> dict` — pod 起動。
  - `runpod.get_pods() -> list[dict]` — 所有 pod 一覧。
  - `runpod.get_pod(pod_id) -> dict` — 個別 pod 詳細。
  - `runpod.stop_pod(pod_id)` — pod を停止 (storage は残る、課金停止)。
  - `runpod.resume_pod(pod_id)` — 停止済み pod を再開。
  - `runpod.terminate_pod(pod_id)` — pod 削除 (storage も消える)。
  - `runpod.get_gpus() -> list[dict]` — GPU 種一覧 (`id`, `displayName`, `memoryInGb`)。
  - `runpod.get_gpu(gpu_id) -> dict` — 特定 GPU の詳細とプライス情報 (`securePrice`, `communityPrice`, `secureSpotPrice`, `communitySpotPrice`, `minimumBidPrice`, `uninterruptablePrice`)。
- **`create_pod` の実引数** (GitHub `runpod/api/ctl_commands.py` から確認):
  ```python
  def create_pod(
      name: str,
      image_name: Optional[str] = "",
      gpu_type_id: Optional[str] = None,
      cloud_type: str = "ALL",            # "ALL" / "SECURE" / "COMMUNITY"
      support_public_ip: bool = True,
      start_ssh: bool = True,
      data_center_id: Optional[str] = None,
      country_code: Optional[str] = None,
      gpu_count: int = 1,
      volume_in_gb: int = 0,              # 一時的な persistent disk
      container_disk_in_gb: Optional[int] = None,
      min_vcpu_count: int = 1,
      min_memory_in_gb: int = 1,
      docker_args: str = "",              # コンテナ起動コマンド (onstart 相当)
      ports: Optional[str] = None,
      volume_mount_path: str = "/runpod-volume",
      env: Optional[dict] = None,
      template_id: Optional[str] = None,
      network_volume_id: Optional[str] = None,
      allowed_cuda_versions: Optional[list] = None,
      ...
  ) -> dict
  ```

### Pod 種別とプライシング

- **Secure Cloud vs Community Cloud** ([Pods overview](https://docs.runpod.io/pods/overview)):
  - Secure: T3/T4 データセンター。安定。Network volume 使用可。
  - Community: P2P 個人ホスト。安価。secure と availability の trade-off。
  - `cloud_type="ALL"` で両方候補 / `"SECURE"` / `"COMMUNITY"` の 3 値。
- **On-demand vs Spot/Interruptible**:
  - Secure / Community ともに on-demand (固定価格、中断なし) と spot (interruptible、入札制) がある。
  - SDK 経由で spot を取るには `bidPerGpu` を指定する `podRentInterruptable` mutation が必要 (Python SDK では `create_pod` から bid 経路は限定的、`runpodctl pod create --bid` の方が柔軟)。
  - **本基盤は on-demand 一択**を推奨 (短時間学習 < 30 分なので中断リスク回避が優先)。
- **課金単位**: per-minute、ingress/egress 無料 ([RunPod Pods overview](https://docs.runpod.io/pods/overview))。
- **GPU 別価格 (2026 年 2 月 [Vast.ai vs RunPod 比較](https://medium.com/@velinxs/vast-ai-vs-runpod-pricing-in-2026-which-gpu-cloud-is-cheaper-bd4104aa591b))**:
  - RTX 4090 24GB: $0.39/hr (Community)
  - RTX 3090 24GB: ~$0.22-0.30/hr (Community)
  - A100 PCIe 40GB: $0.60/hr (Secure)
  - A100 SXM 80GB: $0.79/hr (Secure)
  - L40 40GB: $0.69/hr (Secure)
  - **Vast.ai は同 GPU で平均 20-30% 安い** (RTX 3090 Vast $0.13 vs RunPod $0.22 等)。RunPod は network speed と reliability で勝るとのレビュー多数。

### Onstart / 起動スクリプト機構

- **RunPod に Vast の `onstart_cmd` 直接相当はない** ([Issue #338 docker_args](https://github.com/runpod/runpod-python/issues/338))。代わりに 2 経路:
  1. **`docker_args="..."`**: `create_pod` の引数。コンテナの ENTRYPOINT/CMD を上書き。**注意**: コマンドが exit すると pod が **再起動して再実行** されるため、`runpodctl stop pod $RUNPOD_POD_ID` で自殺するか、末尾に `sleep infinity` を付ける必要 ([Pod restarting answer](https://www.answeroverflow.com/m/1321260714115072000))。
  2. **イメージ内 `/start.sh`** ([Initialization Scripts - DeepWiki](https://deepwiki.com/runpod/containers/6.2-initialization-scripts)): Dockerfile に `ADD start.sh /` + `RUN chmod +x /start.sh` + `CMD ["/start.sh"]` を入れる。`pre_start.sh` / `post_start.sh` のフック点もある。
- **Self-destruct from within the pod** ([Self terminate answer](https://www.answeroverflow.com/m/1267200524181180427)):
  - 全 Pod に `runpodctl` が pre-install される + pod-scoped API key も自動付与。
  - `runpodctl stop pod $RUNPOD_POD_ID` で停止 (volume は残るが課金止まる) / `runpodctl remove pod $RUNPOD_POD_ID` で完全削除。
  - 環境変数 `RUNPOD_POD_ID` が pod 内で必ず設定される (Vast の `VAST_CONTAINERLABEL` 相当)。
- **タイムアウト による自殺** ([AI on a schedule](https://www.runpod.io/articles/guides/ai-on-a-schedule)):
  - `bash -c "nohup sleep 2h; runpodctl stop pod $RUNPOD_POD_ID" &` を docker_args に仕込む例。コスト爆走の保険として有用。

### Network Volume

- **特性** ([Network volumes](https://docs.runpod.io/storage/network-volumes)):
  - 永続化、Pod から独立。NVMe SSD で 200-400 MB/s。
  - 価格: 最初の 1TB は $0.07/GB/月、超過分 $0.05/GB/月。Vast.ai の network volume 平均 $0.10-0.20/GB/月より **2 倍安い**。
  - **Secure Cloud 専用**。Community Cloud Pod には attach 不可。
  - **Pod 作成時に `network_volume_id=` で指定。Pod 起動後の attach/detach は不可** (Pod 削除→再作成が必要)。Vast.ai と同じ制約。
  - 4TB 超は support 連絡。
  - 対応データセンター: US-KS-2 等の特定 DC のみ。**volume が乗っている DC で pod を立てる必要がある** ので、Pod 検索時に `data_center_id` を volume と揃える必要がある。
- **API**: REST / Web UI / `runpodctl network-volume {create|delete|get|list}` ([runpodctl_network-volume](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_network-volume.md))。Python SDK には公式 wrapper が薄く、REST 直叩きが必要 ([Issue #172 runpodctl GraphQL fail](https://github.com/runpod/runpodctl/issues/172))。
- **本基盤での扱い**: 初期版は **手動で 1 個 network volume を作成** → `RUNPOD_NETWORK_VOLUME_ID` を `backend/.env` に書いておく → `runpod train` でその id を `network_volume_id=` で渡す、で十分。自動 search/create は Phase 2 (`volume search` / `volume create` サブコマンド) で対応。

### Logs / 監視

- **CLI**: `runpodctl pod logs <pod_id>` で stdout/stderr 取得。Web UI からも可。
- **SSH**: `start_ssh=True` (デフォルト) + `support_public_ip=True` で SSH 経由デバッグ可能。
- Vast.ai の `vastai logs <id>` 体験と同等。

### REST API / GraphQL

- 2026 年に **新 REST API** がリリース ([REST API blog](https://www.runpod.io/blog/runpod-rest-api-gpu-management))、ただし Python SDK は依然 GraphQL 経由 (`runpod.api.graphql.run_graphql_query`)。本基盤は SDK でのみ呼ぶ。

## Similar OSS Projects

### Project 1 — [SkyPilot](https://github.com/skypilot-org/skypilot)

- **Relevance**: マルチクラウド GPU runner (Vast.ai / RunPod / AWS / GCP / Azure / Lambda Labs を 1 CLI で抽象化)。
- **Approach**: `~/.sky/config.yaml` + `sky launch <task.yaml>` で yaml の `resources: cloud: runpod, accelerators: A100` を解釈し、各クラウド SDK を呼ぶ Plugin パターン。各 cloud は `sky/clouds/runpod.py` 等に provisioning ロジックを実装。`onstart` は yaml の `setup:` / `run:` ブロックで bash 文字列指定。
- **Reusable patterns**:
  - 1 リポ内で複数 GPU プロバイダを扱うときの **provider abstract base class** (`Cloud` / `ResourceHandle`) — provider 増えても CLI top レベルが膨らまない。
  - `setup` (initialization) と `run` (job 本体) を分けて両方 ssh 経由で実行する設計。
- **Pitfalls found**:
  - Issue tracker に「RunPod の docker_args は exit すると再起動するので、idle 化を意識的にやらないと無限再起動する」報告あり (RunPod 共通の罠)。
  - Network volume の region 制約に何度かハマったログがある。
- **Applicability**: 本基盤は当面シングルプロバイダ (RunPod 単体) なので、SkyPilot ほど大袈裟な抽象は不要。ただし「将来 Vast / RunPod 両基盤を `dev/cloud train --provider=vast|runpod` で統一する」の伏線として、`backend/src/runpod_io/` と `backend/src/vast/` を **同じ public interface (search_offers, create_instance, destroy_instance) で揃える** 価値はある。

### Project 2 — [runpod/containers](https://github.com/runpod/containers)

- **Relevance**: RunPod 公式の base Docker image collection。`pre_start.sh` / `start.sh` / `post_start.sh` の hook パターンを実装。
- **Approach**: `pytorch/`, `cuda/`, `vllm/` 等のディレクトリに各 image の Dockerfile + start scripts。`start.sh` で SSH daemon 起動 + Jupyter + ユーザ supplied `pre_start.sh` 呼び出し → `sleep infinity`。
- **Reusable patterns**:
  - 「ユーザは `pre_start.sh` を S3 や git から取得して `/pre_start.sh` に置けばカスタム init できる」設計。Vast の onstart_cmd 同等の柔軟性を user supplied script で確保。
- **Pitfalls found**: 公式 image を使う場合、SSH daemon が SIGTERM をキャッチして遅延終了するケースがあり、self-destroy 後にも数十秒残ることがある (post-mortem ログより)。
- **Applicability**: 本基盤は **base image を `runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04` 等の公式 RunPod PyTorch image に固定** + `docker_args` で onstart スクリプトを渡す方式が最小差分。`pre_start.sh` フックは不要 (我々の起動コードは `docker_args` の中で完結)。

### Pattern Comparison

| 観点 | 本基盤 (vast 既存) | SkyPilot | runpod/containers |
|------|------|----------|------|
| 抽象レベル | 1 プロバイダ専用 | マルチクラウド | image only |
| Onstart | onstart_cmd (Vast SDK) | yaml `setup:` + `run:` | `pre_start.sh` hook |
| Self-destroy | `vastai destroy` (CLI 内) + EXIT trap | プロビジョニング系で teardown | image 内で `runpodctl stop pod` を user 任せ |
| Volume | network volume (search/create/attach) | provider 抽象 | image only |
| Recommendation | **MVP は独立 mirror** (vast を真似て runpod を作る) | Phase 2 で参考 (provider abstract) | image 戦略の参考 (公式 pytorch image を流用) |

## Library/Service Selection

### Python SDK

| Candidate | Pros | Cons | Maintenance | Recommendation |
|-----------|------|------|-------------|----------------|
| ⭐ `runpod>=1.7` (公式) | ✅ 公式、SDK 主要 API カバー / ✅ env / network_volume / docker_args 対応 / ✅ `get_gpus()` で GPU 種列挙可 | ⚠️ network volume CRUD は薄い、REST 直叩きが必要なケースあり / ⚠️ spot 入札は限定的 | アクティブ。週次 release。 | **採用**。on-demand 学習なら現行機能で十分。 |
| `pulumi-runpod-native` (community) | ✅ Terraform 風の宣言的 IaC / ✅ 状態管理 | ⚠️ 学習コスト高、本基盤は ephemeral pod なので IaC 不要 | community | 不採用 |
| GraphQL 直叩き | ✅ 全機能アクセス可 | ⚠️ boilerplate 多い、SDK と乖離する maintenance 負債 | DIY | 不採用 |

💡 推薦理由: 既存 vast 基盤と同じ「公式 SDK ラッパ」スタイルで mirror すれば、コードレビュー負担と保守コストが最小。

### CLI: `runpodctl` ローカルインストールは不要

- Pod 内には pre-install 済みなので、self-destroy には container 内の `runpodctl stop pod` が使える。
- ローカル CLI 操作 (`dev/runpod`) は Python SDK 経由で完結させる方針 (Vast 基盤と一致)。

## API/Protocol Research

### `runpod.create_pod` 呼び出し例 (本基盤想定)

```python
import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]

pod = runpod.create_pod(
    name=f"orbit-wars-{run_id}",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 3090",   # get_gpus() で取得した id
    cloud_type="SECURE",                       # network volume 使うなら必須
    gpu_count=1,
    container_disk_in_gb=40,
    volume_in_gb=0,                            # network volume 使うので一時 disk は 0
    network_volume_id=os.environ.get("RUNPOD_NETWORK_VOLUME_ID"),
    volume_mount_path="/persist",              # vast 既存 onstart と同じ mount path
    docker_args=f"bash -c '{onstart_script}'",  # 1 行に折り畳んだ onstart bash
    env={
        "AWS_ACCESS_KEY_ID": ...,
        "AWS_SECRET_ACCESS_KEY": ...,
        "AWS_DEFAULT_REGION": "ap-northeast-1",
        "ORBIT_WARS_RUN_ID": run_id,
        "ORBIT_WARS_GIT_SHA": commit_sha,
        "ORBIT_WARS_GIT_BRANCH": branch,
        "ORBIT_WARS_CASE": case,
    },
    support_public_ip=True,
    start_ssh=True,
    ports="22/tcp,8888/http",
)
pod_id = pod["id"]
```

### `runpod.get_gpus()` で GPU 種を列挙 + filter

```python
gpus = runpod.get_gpus()
# [
#   {"id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24},
#   {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24},
#   {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "memoryInGb": 48},
#   ...
# ]

# 価格付きでさらに詳細取得
detail = runpod.get_gpu("NVIDIA GeForce RTX 3090")
# {
#   "id": ..., "displayName": ..., "memoryInGb": 24,
#   "secureCloud": True, "communityCloud": True,
#   "securePrice": 0.43, "communityPrice": 0.22,
#   "secureSpotPrice": 0.18, "communitySpotPrice": 0.11,
#   "minimumBidPrice": 0.11, "uninterruptablePrice": 0.22,
#   "lowestPrice": {"minimumBidPrice": 0.11, "uninterruptablePrice": 0.22, ...}
# }
```

### Self-destroy (onstart 末尾)

```bash
# RunPod は runpodctl pre-install + pod-scoped API key 自動設定済み
echo "[onstart] self-destroy pod=${RUNPOD_POD_ID}"
runpodctl stop pod "${RUNPOD_POD_ID}"
```

Vast 版では Python SDK 呼び出しでの self-destroy (venv 内 vastai SDK) が必要だったが、RunPod では **`runpodctl` CLI 1 行で済む** ためテンプレが短くなる。

## Research Summary

- **Vast.ai 基盤の構造をそのまま mirror** して RunPod 基盤を作るのが最小差分・最大流用。`backend/src/runpod_io/` を新設、`backend/src/vast/` の各ファイルに 1:1 対応する `auth.py` / `offers.py` / `instance.py` / `run_meta.py` / `cost.py` / `cli.py` / `onstart.sh.tmpl` / `__main__.py` を実装。
- **provider 抽象は MVP では作らない**。両基盤が同居する場合、`run_meta.py` の `vast_instance_id` フィールドを 2 つに分ける (vast 側は維持、runpod 側に `runpod_pod_id` を追加) の最小拡張で済ませる。共通化は Phase 2。
- **Onstart 機構** は `docker_args` に bash 文字列を渡す方式が clean。Vast の `onstart_cmd` と概ね等価。末尾で `runpodctl stop pod $RUNPOD_POD_ID` で自殺、SDK install 不要なので Vast 版より bash テンプレが **短く** なる。
- **Network volume** は Secure Cloud 専用 + 作成時 attach のみ。MVP は手動で 1 個作って `backend/.env` に id を書く。`volume_in_gb=0` + `network_volume_id` で attach、`volume_mount_path="/persist"` で Vast と同じ mount path 規約。
- **GPU 検索** は SDK の `get_gpus()` + 各 id ごとに `get_gpu(id)` で価格を取り、(`secureCloud` 真 / `securePrice <= max_price` / `memoryInGb >= 16` / 等) フィルタで絞ってから rich Table 表示 + 数字選択。Vast の `search_offers(query=...)` の DSL 構文より処理が散らばるので、`offers.py` 内で **caching と filter pipeline** を組む。
- **Cost report** は `runpod_offer_snapshot.dph_total × runtime_seconds / 3600` で同じ式を再利用 (`vast.cost.aggregate_runs` → `runpod.cost.aggregate_runs` に内容コピー、メッセージ文字列のみ差分)。出力 path は `docs/experiment/runpod_cost_report_<YYYY-MM>.md`。
- **Recommended approach**:
  1. `backend/src/runpod_io/` を新規作成 (vast を mirror)。
  2. `dev/runpod` thin wrapper (`exec uv run --directory backend python -m runpod_io "$@"` — `runpod` モジュール名は SDK と衝突するので **パッケージ名は `runpod_io`** とする (ユーザ確定))。
  3. `train.py` 側は **無改修**。`ORBIT_WARS_RUN_DIR` 等の既存 env は provider 中立なのでそのまま使える。`run_meta.RunMetadata` には `runpod_pod_id: int | None` フィールドを追加 (vast 側は None のまま、後方互換確保)。
  4. `data/output/models/imitation/case<N>/runs/<run_id>/` の run dir scheme は両基盤で共有。`run_id` 内に provider 識別子を入れない (commit-sha と timestamp で衝突回避)。
  5. e2e は手動で 1 度だけ実行 (Vast と同じ運用)。

## Sources

- [Manage Pods - Runpod Documentation](https://docs.runpod.io/pods/manage-pods)
- [GitHub - runpod/runpod-python](https://github.com/runpod/runpod-python)
- [Pod Creation and Configuration | DeepWiki](https://deepwiki.com/runpod/docs/3.1-pod-creation-and-configuration)
- [Network volumes - Runpod Documentation](https://docs.runpod.io/storage/network-volumes)
- [runpodctl GitHub](https://github.com/runpod/runpodctl)
- [Streamline GPU Cloud Management with RunPod's New REST API](https://www.runpod.io/blog/runpod-rest-api-gpu-management)
- [Vast.ai vs RunPod pricing in 2026](https://medium.com/@velinxs/vast-ai-vs-runpod-pricing-in-2026-which-gpu-cloud-is-cheaper-bd4104aa591b)
- [RunPod vs Vast.ai 2026 Comparison](https://valebyte.com/en/guides/runpod-vs-vastai-real-llm-inference-benchmarks-cost-analysis/)
- [Issue #338: docker_args restart behavior](https://github.com/runpod/runpod-python/issues/338)
- [Possible to terminate pod from within the pod](https://www.answeroverflow.com/m/1267200524181180427)
- [AI on a Schedule: RunPod auto-stop](https://www.runpod.io/articles/guides/ai-on-a-schedule)
- [Initialization Scripts | runpod/containers DeepWiki](https://deepwiki.com/runpod/containers/6.2-initialization-scripts)
