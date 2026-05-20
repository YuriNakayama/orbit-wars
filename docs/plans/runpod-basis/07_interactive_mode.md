# Interactive Mode (dev / debug pods)

`dev/runpod dev` で起動する RunPod pod は **インタラクティブモード** で動作する。
従来の `dev/runpod train` (oneshot モード) と異なり、自動 terminate を行わず、
SSH 接続でコード変更・再実行・デバッグを繰り返すための運用形態。

## モード比較

| 項目 | oneshot (`train`) | interactive (`dev`) |
|------|-------------------|---------------------|
| onstart 後の挙動 | preprocess → train → DVC push → 自動 remove | preprocess + uv sync → `sleep infinity` で待機 |
| 8h timeout guard | あり (`runpodctl remove pod`) | **なし** |
| `trap cleanup_destroy EXIT` | あり | **なし** |
| 終了方法 | 自動 | `dev/runpod destroy <run_id>` で明示 |
| `launch.json.mode` | `"oneshot"` | `"interactive"` |
| 進捗マーカ完了サイン | `99_done` | `50_interactive_ready` |
| 用途 | CI / 学習バッチ | コード試行錯誤・デバッグ |

切り替えは内部的に `render_onstart(..., mode="interactive")` が `<RUNPOD_MODE>`
プレースホルダを置換することで実現。`20_cleanup.sh` と `60_train.sh` の冒頭で
`if [ "${RUNPOD_MODE}" = "oneshot" ]` 条件分岐している。

## 典型ワークフロー

```bash
# 1) ブランチを push して dev pod を立てる
git push origin feature/foo
dev/runpod dev <commit-sha> --case case1

# → 出力:
#   Interactive pod launched! id=pod_abc run_id=20260520-...
#   Wait for setup:    dev/runpod status <run_id> --case case1
#   Open SSH:          dev/runpod ssh <run_id> --case case1
#   Sync code:         dev/runpod sync <run_id> --case case1 --push
#   Destroy when done: dev/runpod destroy <run_id> --case case1

# 2) onstart の uv sync / DVC pull が完了するまで待機
dev/runpod status <run_id> --case case1
#   progress: latest=50_interactive_ready ...

# 3) SSH で接続 (proxy 経由がデフォルト、Community Cloud でも動く)
dev/runpod ssh <run_id>
#   → root@pod-xxx:/workspace/orbit-wars/bot#

# 4) ローカルでコード修正したら push
dev/runpod sync <run_id> --push --dry-run    # まず diff 確認
dev/runpod sync <run_id> --push

# 5) pod 上で再実行 (ssh exec での 1-shot 実行も可)
dev/runpod ssh <run_id> --exec "cd /workspace/orbit-wars/bot && uv run python -m pipeline.imitation.case1.training.train --config pipeline/imitation/case1/configs/il.yaml"

# 6) 終わったら忘れず destroy (interactive は課金停止しないと止まらない)
dev/runpod destroy <run_id>
```

## SSH 経路: proxy vs direct

| 経路 | コマンド形 | 既定 key | 適合場面 |
|-----|-----------|---------|---------|
| **proxy** (既定) | `ssh <pod_id>@ssh.runpod.io -i ~/.ssh/id_ed25519` | `~/.ssh/id_ed25519` | Community Cloud 含む全環境で安定 |
| **direct** | `ssh root@<ip> -p <port> -i ~/.runpod/ssh/RunPod-Key-Go` | `~/.runpod/ssh/RunPod-Key-Go` | TCP/22 公開 port 経由、Secure Cloud |

切替は `dev/runpod ssh --via proxy|direct`、`dev/runpod sync --via proxy|direct`。
proxy SSH を使うには `~/.ssh/id_ed25519.pub` を <https://runpod.io/console/user/settings>
にあらかじめ登録しておく必要がある。

## 課金停止の安全弁

- `dev/runpod ps` は interactive pod の `mode` 列を **黄色** で表示し、末尾に
  destroy リマインダを出す。
- `dev/runpod destroy` は確認プロンプト付き (skip するなら `-y`)。
- 8h timeout guard が **無い** ため、放置すると 24h × $0.5/h = $12/day などが
  止まらない。stop し忘れに注意。

## rsync の挙動

`dev/runpod sync` は内部で `runpod_io.runpod.sync.build_sync_plan` を使い、
以下を必ず exclude する (`DEFAULT_EXCLUDES`):

- `.venv/`, `.venv.*/` — Linux/Mac mismatch を防ぐ
- `data/` — DVC 管理
- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `*.egg-info/`, `.DS_Store`

`--delete` は `--dry-run` と組み合わせる運用を推奨 (確認プロンプトあり)。

## 既知の限界

- `dev` mode の pod は `dev/runpod watch` の対象外 (`99_done` に到達しないため永久に
  pending と見なされる)。状態確認は `dev/runpod status` を使う。
- `dev/runpod cost-report` は interactive と oneshot を分離集計しない。月次レポート
  で interactive 分を区別したい場合は `runs/<run_id>/launch.json.mode` を grep する。
- pod 内で git commit してから `--pull` した場合、ローカル側の `.git` と衝突する可能性
  がある。コード反映は片方向 (`--push` only) を推奨。
