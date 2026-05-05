# imitation/case0 — RunPod E2E 検証

## 背景

過去の RunPod 試行では、実際の学習に到達するまでに以下のような落とし穴で複数回失敗している (`memory/runpod_5_traps_2026_05_04.md`, `memory/runpod_onstart_pitfalls.md`)。

- `dvc pull` が他 case の outs まで巻き込み time / cost 爆発
- mart_dvc symlink 切れ
- mark_progress 欠落で `dev/runpod logs` が空になる
- cuda 13 driver mismatch
- RTX 4090 ノードガチャ (image pull stuck / scheduling 失敗)
- onstart の cwd-relative config path

これらは「学習ロジックを書く前に基盤が壊れている」状態であり、実モデルを走らせて初めて気付く設計になっていた。case0 は **学習ロジックを完全に切り離した最小 pipeline** を作り、RunPod 基盤そのものを E2E で検証する場とする。

## 目的 (スコープ)

RunPod GPU 基盤 (`dev/runpod`) が以下を満たすことを 1 回の run で検証可能にする:

1. ローカル smoke test → push → `dev/runpod train` → onstart → 学習プロセス → S3 marker → DVC/S3 artifacts → `dev/runpod pull` の全経路が GREEN になる
2. 実行ステップ単位の構造化ログ・進捗 marker が可視化される
3. CPU / RAM / GPU の使用量が `dev/runpod tail --source {gpu,system}` から live tail できる
4. 失敗時には pod が確実に terminate され、retryable な失敗 (scheduling / image pull / driver) は新規 run_id で自動 retry される

学習アルゴリズム自体の改善はスコープ外。case0 のモデルは ~200 パラメータの TinyMLP で、合成データ上を 1 epoch / 10 step で完走する。

## 仮説

**仮説**: 上記 6 つの落とし穴を構造的に排除し、smoke test を pre-flight gate にすれば、RunPod E2E run は cost < $0.20 / runtime < 15min で 1 発成功する。

**検証方法**: 実 run を 1 本走らせ、Definition of Done (D1–D12) を満たすこと。

## 設計

### 1. case0 ディレクトリ構成

```
bot/pipeline/imitation/case0/
├── __init__.py
├── main.py                           # Path.cwd() pattern (Kaggle submit 互換)
├── README.md
├── policy/
│   ├── __init__.py
│   ├── agent.py                      # NO_OP stub
│   └── model.py                      # TinyMLP (Linear 8→16→4, ~200 params)
├── training/
│   ├── __init__.py
│   ├── dummy_data.py                 # deterministic 合成 parquet 生成
│   ├── dataset.py                    # TensorDataset 薄ラッパ
│   └── train.py                      # 1 epoch / max_steps=10 の最小 train loop
├── evaluation/
│   ├── __init__.py
│   └── eval_smoke.py                 # 1 match (env.run([agent, "random"]))
└── configs/
    └── smoke.yaml                    # num_samples=64, batch=4, max_steps=10
```

### 2. RunPod 基盤への追加 / 変更点

| 変更 | 内容 | 理由 |
|------|------|------|
| `runpod_io/system_monitor.py` 新規 | psutil で CPU% / RAM / load avg を 10s 周期で `system.log` へ JSON-line 出力 | nvidia-smi (gpu.log) は GPU しか取れない。OOM 痕跡や CPU 過負荷も観測できるよう常駐 sampler を追加 |
| `runpod_io/progress.py` 拡張 | `write_progress_marker(step)` を追加 | bash の `mark()` だけだと粗い (data_load / model_init / step / eval / save の境界が打てない)。Python 側からも S3 marker を push 可能に |
| `runpod_io/onstart.sh.tmpl` 更新 | `nvidia-smi` と並列で `system_monitor` を background 起動、終了時 SIGTERM で停止 | system.log を生成 |
| `runpod_io/cli.py` `tail` | `--source system` を追加 | system.log を SSH 経由 live tail |
| `runpod_io/cli.py` `train` | `--skip-smoke` / `--dry-run` を追加、デフォルトでは `_run_preflight_smoke(case)` で **case0 CPU smoke + import sanity** を gate | コミットの import error / yaml 壊れを **RunPod を起動する前に** 検出。1 ヶ所の typo で $1.5 が飛ぶのを防ぐ |
| `runpod_io/cleanup.py` 新規 | `terminate_pod(sdk, pod_id)` (idempotent) | bash trap が走らない経路 (image pull stuck / kernel SIGKILL) で billable seconds が止まらないリスクを潰す。watcher が failure を返したら必ず呼ぶ |
| `runpod_io/retry.py` 新規 | `FailureReason` enum + `decide(...)` policy + `RetryChain` | retryable な失敗 (scheduling / image / driver) を新規 run_id で自動再起動。OOM / non-zero exit は retry しない (無駄遣い回避) |
| `case0` 用 CASE_DEFAULTS | train_module / config_arg を登録 | `dev/runpod train --case case0` を有効化 |

### 3. 落とし穴ごとの対応マッピング

| 落とし穴 (memory) | case0 での対応 |
|------------------|----------------|
| dvc pull other case outs | case0 用に `data/lake/case0_smoke/dvc_smoke_marker.txt` (55 bytes) を新設。`onstart.sh.tmpl` の `<CASE>=case0` 分岐で `dvc pull data/lake/case0_smoke.dvc` のみ実行し、他 case の outs は触らない (memory: runpod_5_traps の最大要因) |
| mart_dvc symlink 切れ | case0 は data/mart に依存しない。symlink 不在時も影響を受けない |
| mark_progress 欠落 | train.py の各境界 (00_data_load / 10_model_init / 20_train_start / 30_train_step_NNNN / 40_eval_start / 50_save / 99_done) で `_mark_progress()` を強制呼び出し |
| cuda 13 driver mismatch | Phase 5 で onstart 早期に `python -c "import torch; assert torch.cuda.is_available()"` を実施。CASE_DEFAULTS に default image を `cu121` 系統で固定する |
| RTX 4090 ノードガチャ | Phase 5 で case0 のデフォルトを `RTX A4000 / RTX 4000 Ada` 系列・cost_limit_usd=0.20 に絞る |
| cwd-relative config path | `_resolve_config_path(config)` を追加: 絶対 path / `<repo>/bot/<config>` / `absolute_under_repo` の順で解決し `Path.cwd()` 依存を排除 |

### 4. ローカル smoke gate

`dev/runpod train --case <any>` 起動時、デフォルトで以下を順に実施:

1. `uv run --directory bot python -c "import {train_module}"` — import sanity
2. `uv run --directory bot python -m pipeline.imitation.case0.training.train --device cpu` — 60s timeout 内で完走

どちらかが失敗すれば exit 1 で **API 呼び出しなしに** ブロックする。`--skip-smoke` で迂回可能 (CI mock 用)。`--dry-run` は smoke + git/aws 検証だけ走らせて pod を起動しない。

### 5. 失敗時の retry policy

```
attempts_so_far  reason                    decide
1                scheduling_failed         retry (rid_2 を新規発行)
1                image_pull_stuck          retry (rid_2)
1                driver_mismatch           retry (rid_2)
1                non_zero_exit / oom       NO retry (user code bug を疑う)
1                cost_limit_hit            NO retry (会計上止める)
3                scheduling_failed         NO retry (max_retries=2 hard cap)
```

retry は新規 `run_id` で再起動する。理由は (a) S3 marker / artifact prefix の衝突回避、(b) 失敗 run の audit trail 保全、(c) ノードガチャ回避には新スケジュールが必要。`run.json.retry_of = <previous_run_id>` で linked list を形成する (Phase 6 のテスト済 helper `RetryChain`)。

## 実装 Phase

| Phase | 内容 | 状態 |
|-------|------|------|
| 1 | case0 directory scaffold + dummy data | done |
| 2 | minimal train.py (CPU 1 epoch / 10 step) | done |
| 3 | system_monitor + progress writer + tail --source system | done |
| 4 | local smoke test + `dev/runpod train` pre-flight gate (`--dry-run` / `--skip-smoke`) | done |
| 5 | RunPod E2E 実行 (ユーザー承認必須) | **pending** — 承認後に実施 |
| 6 | cleanup.py / retry.py + watcher 失敗時の自動 terminate | done (logic + tests; Phase 5 結果で追加チューニング) |
| 7 | この plan.md + Phase 5 後に result.md | in progress |

## 完了条件 (Definition of Done)

| # | 条件 | 検証方法 |
|---|------|----------|
| D1 | case0 が CPU で 90s 以内に train smoke 完走 | `time uv run python -m pipeline.imitation.case0.training.train --device cpu` |
| D2 | `dev/test-bot` 緑 (case0 test 含む)、既存 case1〜7 のテスト無破壊 | pytest |
| D3 | `dev/runpod train --case case0 --dry-run` が smoke を強制実行し、smoke 失敗時 exit 1 | 故意に train.py を壊して再現 |
| D4 | 実 RunPod run で `99_done` marker 到達、cost < $0.20、runtime < 15min | `dev/runpod summary <run_id>` |
| D5 | marker timeline が `00 → 10 → 20 → 30_* → 40 → 50 → 99` の順に揃う | `dev/runpod logs <run_id> --source markers` |
| D6 | `dev/runpod tail --source {train,gpu,system,onstart}` の 4 経路が live 出力 | 手動確認 (Phase 5) |
| D7 | 故意失敗 (bad image) で auto cleanup → 新規 run_id で retry → 2 回目成功 | `dev/runpod summary <retry_run_id>` の `retry_of` 確認 |
| D8 | retry が `max_retries=2` で hard cap される | unit test (`tests/src/runpod_io/test_retry.py`) |
| D9 | `run.json` に `failure_reason` が enum 値で記録される | mock test + 実 run の手動確認 |
| D10 | `docs/experiment/imitation/20260505_case0_runpod_e2e/{plan.md,result.md}` 存在 | `ls` |
| D11 | memory に記録された 6 trap を case0 で踏まないことを result.md にチェックリストで記録 | result.md |
| D12 | `bot/.env` 等 secrets を一切読まない / 触らない | grep `\.env` がゼロ |

D1–D3, D8 は実装で達成済み (Phase 1〜6)。D4–D7, D9, D11 は Phase 5 (RunPod 実行) で確認する。

## Phase 5 起動パラメータ (確定)

| パラメータ | 値 | 適用方法 |
|-----------|----|----------|
| `--cloud-type` | `SECURE` (default) | network volume attach のため |
| `--volume-name` | `orbit_wars` (default) | 既存 volume を再利用 (Secure 必須) |
| `--mount-path` | `/persist` (default) | 既存運用と整合 |
| GPU | **RTX 3090 単独** | `cli.py` の `case == "case0"` 分岐で `gpu_names = ["NVIDIA GeForce RTX 3090"]` に絞る (4090 ガチャ回避 + 最安) |
| `--max-dph` | **0.4** | 同上分岐で default 2.0 から自動縮小 |
| `--cost-limit` | **$0.30** | 同上分岐。DVC pull ハンドシェイクを含むので default $0.20 より緩める |
| `--watch` | 推奨 | デスクトップ通知 + 自動 cleanup |
| DVC pull | **scoped** | `onstart.sh.tmpl` の `<CASE>=case0` 分岐で `dvc pull data/lake/case0_smoke.dvc` のみ (55 bytes) |

ユーザーが `--gpu-name` / `--max-dph` / `--cost-limit` を明示すればその値が優先される (default 値のときだけ縮小ロジックが効く)。

## NEXT ACTION

ユーザー承認を得てから:

1. 作業ブランチを push: `git push origin feature/runpod-repair`
2. RunPod 起動: `dev/runpod train <commit-sha> --case case0 --watch`
3. 完走 / 失敗ログを `dev/runpod summary <run_id>` / `dev/runpod logs <run_id>` で回収
4. 計測結果を `result.md` に書き込み、D4–D7, D9, D11 のチェックを埋める
5. `result.md` を同 PR で commit
