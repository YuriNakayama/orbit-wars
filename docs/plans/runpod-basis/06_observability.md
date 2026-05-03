# Observability — RunPod 学習基盤の監視と成果物管理

RunPod 公式 API には pod logs エンドポイントが無い ([runpod-python#400](https://github.com/runpod/runpod-python/issues/400) で要望中、未実装) ため、本基盤では **S3 を経由した非同期チャネル** と **SSH 経由のライブ tail** を組み合わせて観測性を確保する。要件は 4 つに分けて整理し、それぞれ permanently / live で別経路を取る。

## 要件マトリクス

| 要件 | 永続化 | 経路 | コマンド |
|------|---|------|------|
| 立ち上げ進捗 (image pull / apt / uv sync / dvc pull) | live | SSH `tail -F /var/log/onstart.log` | `dev/runpod tail <id> --source onstart` |
| 学習中の epoch / loss / GPU 使用率 | live | SSH `tail -F train.log` / `gpu.log` | `dev/runpod tail <id> --source train\|gpu` |
| 完了後の status / コスト / 所要時間 / 最終 metrics | persistent | S3 markers + run.json + launch.json を集約 | `dev/runpod summary <id>` |
| 成果物 (best.pt / metrics.json / run.json / onstart.log) | persistent | DVC + S3 artifacts prefix の二重化 | `dev/runpod pull <id>` (auto fallback) |

## ライブ監視 (S3 不要)

### `dev/runpod tail <run_id> --source <X>`

pod が RUNNING の間だけ機能する SSH ベースの実況。`onstart.sh.tmpl` 冒頭で sshd を早期起動してあるため、container 起動後数秒で接続可能。

- `--source onstart` → `tail -F /var/log/onstart.log` (立ち上げ + 全体)
- `--source train` → `tail -F data/output/.../runs/<id>/train.log` (学習プロセス stdout)
- `--source gpu` → `tail -F data/output/.../runs/<id>/gpu.log` (`nvidia-smi -l 10` のサンプル)
- `--no-follow` → `tail -n 200` で末尾 200 行だけ取って exit

`launch.json` (ローカル) から pod_id を解決し、`runpod_io.ssh.get_pod_ssh_endpoint` で host:port を取得して `ssh -i ~/.runpod/ssh/RunPod-Key-Go ...` を fork する。

terminate 後はこの経路は使えない (SSH 接続不可)。代わりに `dev/runpod logs --source onstart` (S3 fallback) を使う。

## 永続的な監視 (S3 経由)

### `dev/runpod summary <run_id>`

複数 source を merge した完成状態。pod 終了後でも S3 さえ生きていれば取れる。

- launch.json (ローカル) — pod_id / dph_total / data_center_id
- S3 markers (`runpod_progress/<id>/`) — 直近 step、経過時間
- S3 artifacts (`runpod_artifacts/<id>/`) — best.pt / metrics.json / run.json / onstart.log の存在確認
- run.json (ローカル or S3 artifacts) — final_train_loss / final_val_loss / epochs_run
- `data/output/.../<id>.dvc` (Git origin) — DVC meta が main に commit されているか

判定:
- `succeeded`: marker `99_done` あり
- `failed`: marker `90_cleanup_exit_<非ゼロ>` あり
- `running`: marker はあるが終了 marker 無し
- `stalled`: `--watch` 中の動的判定 (15 分以上 marker 更新なし)

### `dev/runpod logs <run_id> --source onstart`

`/var/log/onstart.log` の S3 snapshot を表示。2 経路のフォールバック:
1. **`run_dir/onstart.log`** (DVC pull 後にローカルにある場合)
2. **`s3://.../runpod_progress/<id>/onstart.log`** (cleanup_destroy / 2h timeout / 5s ストリーマがアップロード)

bash が動き始めれば必ずどこかにあるはず。bash が一度も走らなかった失敗 (image pull stuck 等) は取得不可。

## 成果物の管理 (二重化)

### S3 直 upload (DVC の保険)

train 完了直後、`dvc add` の **前** に `aws s3 cp` で以下を `s3://orbit-wars-dvc-286854171013/runpod_artifacts/<RUN_ID>/` 配下へ直接 upload:

- `best.pt`
- `metrics.json`
- `run.json`
- `onstart.log`

これにより `dvc push` / `git push <RUN_ID>.dvc` が間に合わずに pod kill されても、artifacts prefix から個別にダウンロードできる。

### `dev/runpod pull <run_id> --from <auto|dvc|s3>`

- `auto` (default): まず DVC 経路を試す → `<RUN_ID>.dvc` が origin に無ければ S3 fallback
- `dvc`: DVC のみ。fallback しない
- `s3`: S3 artifacts prefix から強制取得

S3 fallback で取得した成果物は **DVC 管理外** なので、後で `<RUN_ID>.dvc` が origin に到達したら `--from dvc` で取り直すか、ローカルで `dvc add` し直す運用とする。

### 完了通知

`dev/runpod train --watch` または `dev/runpod watch <run_id>` で、pod の終端を検知して macOS / Linux のデスクトップ通知を発火 (`runpod_io.notify`)。outcome:
- `success` — `99_done` または `90_cleanup_exit_0`
- `failure` — `90_cleanup_exit_<非0>` / pod が EXITED/TERMINATED で 99_done 未到達
- `stalled` — 15 分間 marker 進捗なし
- `timeout` — `--max-wait` 超過 (デフォルト 2h)

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| `bot/src/runpod_io/onstart.sh.tmpl` | sshd 早期起動、tee/nvidia-smi、early artifact upload、5s log streaming |
| `bot/src/runpod_io/ssh.py` | `runtime.ports` から SSH 接続情報を解決 |
| `bot/src/runpod_io/progress.py` | S3 markers + onstart.log + artifacts の reader |
| `bot/src/runpod_io/summary.py` | 全 source merge → `RunSummary` |
| `bot/src/runpod_io/watcher.py` | poll → success/failure/stalled/timeout 判定 + 通知 |
| `bot/src/runpod_io/cli.py` | `tail` / `summary` / `pull --from` サブコマンド |

## 制約

- bash が一度も走らない失敗 (image pull / scheduling / host 死亡) は **観測手段なし**。RunPod 公式 API の制約のため。Web UI からのみ確認可能、terminate 後は消失。
- Community Cloud は SSH の port 公開仕様が異なる場合あり。SECURE Cloud のみ動作確認済み。
- S3 PUT は 5s ストリーミング + 30s heartbeat で 1 run あたり数百回 (~$0.005)。無視できる。
