# kaggle-kernel-basis — Testing Strategy

> **Implementation status (2026-05-20)**: Phase 1-4 完了 + 全 9 case (case1/3/4/5/6/7/8/9/10) の train.py に env パッチ適用済。Unit + integration テスト 76 件 green (kaggle_kernel) + vast/runpod/imitation 全 case で 507 件 green。
>
> **E2E smoke 実施 (2026-05-20, in progress)**: case1 で実 Kaggle 上に dataset (`yurinakayama/orbit-wars-bot`) + kernel を push。下記の実機で発覚した issue を順次潰した:
> - `dir_mode='zip'` がデフォルト `'skip'` のため、bot/ / simulator/ / data/ のサブディレクトリが upload されない (修正: `dir_mode='zip'` を api 呼び出しに追加)
> - Kaggle SDK `kernels_push_cli(folder, timeout, acc)` が timeout/acc 位置引数必須 (修正: `(str(path), None, None)` で呼ぶ)
> - kernel slug は metadata.json の `id` ではなく `title` 由来でスラッグ化される (修正: title と slug を同じ basename から導出)
> - private dataset の mount path は `/kaggle/input/<slug>` ではなく `/kaggle/input/datasets/<owner>/<slug>` (修正: cell B に dataset mount auto-discovery を追加)
> - `find_repo_root()` が `(bot/, .git/)` ペアを探すため `.git/HEAD` プレースホルダが必要 (修正: snapshot builder で `dest_dir/.git/HEAD` を作成)
> - `pip install -e .` が pyproject の `[tool.uv.sources]` で失敗 (修正: pip でなく sys.path 注入で bot 修飾子を importable に)
> - SDK 1.7+ は `kernels_status` が dict ではなく `ApiGetKernelSessionStatusResponse` object を返し、`.status` は `KernelWorkerStatus` Enum (修正: `parse_status` が `.name` 属性を読み、`str(...)` の `KernelWorkerStatus.X` repr も処理する)


## Test Pyramid

```
                    ┌───────────────────────┐
                    │ E2E smoke (Step 10)   │  ← 手動、本番 Kaggle で 1 case
                    │ 1 run / commit        │
                    └───────────────────────┘
                  ┌─────────────────────────────┐
                  │ Integration (Step 8)        │  ← CLI 統合、KaggleApi 全 mock
                  │ python -m kaggle_kernel ... │
                  └─────────────────────────────┘
              ┌────────────────────────────────────┐
              │ Unit (Step 1-7)                    │  ← 各 module 単位、pytest
              │ auth / dataset / template / runner │
              │ artifacts / cost                   │
              └────────────────────────────────────┘
```

## Unit Tests

### `test_run_meta.py` (Step 1)
- 既存 vast/runpod run.json fixture を read → `kaggle_kernel_meta` が default `None`
- `kaggle_kernel_meta = {"accelerator": "gpu-t4x2"}` で write → read で値保持
- 三 provider field を全て埋めた run.json を round-trip

### `test_train_run_dir.py` (case1/case3/case4/case8/case9, Step 2)
- `ORBIT_WARS_KAGGLE_KERNEL_SLUG="user/foo"` 単独 → `kaggle_kernel_meta` 埋まり、他 provider field は `None`
- vast + kaggle_kernel 同時 set → `RuntimeError`、メッセージに 3 経路名前
- runpod + kaggle_kernel 同時 set → `RuntimeError`
- vast + runpod + kaggle_kernel 全 set → `RuntimeError`
- `ORBIT_WARS_KAGGLE_KERNEL_META='{"accelerator":"gpu-t4x2"}'` で展開、空 string で `None`
- malformed JSON で `RuntimeError`

### `test_auth.py` (Step 3)
- env 経路: `KAGGLE_USERNAME` + `KAGGLE_KEY` set → `KaggleCreds` 返却
- `.env` 経路: env 未設定 + `bot/.env` に値 → `KaggleCreds` 返却
- `~/.kaggle/kaggle.json` 経路: env / .env 未設定 + json 存在 → `KaggleCreds` 返却
- 3 経路全て不在 → `CredentialsError` (メッセージに 3 経路の hint 含む)
- `kaggle.json` malformed JSON → `CredentialsError`

### `test_dataset_builder.py` (Step 4)
- snapshot に `data/` が含まれない
- snapshot に `.git/` / `__pycache__/` が含まれない
- `bot/src/`, `bot/pipeline/`, `bot/pyproject.toml`, `simulator/python/`, `simulator/rust/src/` が含まれる
- `include_wheels=[Path("x.whl")]` で `<dest>/wheels/x.whl` に配置される

### `test_dataset_api.py` (Step 4)
- `push_dataset_version` が `KaggleApi().dataset_create_version` を正しい引数で呼ぶ
- `create_new_dataset` が `dataset_create_new` を呼ぶ
- `latest_version_commit` が `version_notes="commit=abc1234"` から `"abc1234"` を抽出
- `dataset_status` が `{"status": "ready"}` を返す mock を round-trip

### `test_kernel_template.py` (Step 5)
- 生成 notebook が `nbformat.validate` を pass
- placeholder 全展開: `<RUN_ID>`, `<COMMIT_SHA>`, `<CASE>`, `<TRAIN_MODULE>`, `<CONFIG_ARG>`, `<DATASET_SLUG>`, `<DATASET_VERSION>`, `<ACCELERATOR>` が残らない
- shell injection 試行 (`run_id=";rm -rf /"`) で `ValueError`
- cell 数が 5 (A,B,C,D,E) または 6 (cleanup 含む)
- cell A の env 一覧に `KAGGLE_KEY` / `AWS_*` が含まれない (secret 漏洩防止)

### `test_kernel_runner.py` (Step 6)
- `push_kernel` が `KaggleApi().kernels_push_cli` を folder path で呼ぶ
- `KernelPushResult` に slug / version が入る
- `poll_status` が QUEUED → RUNNING → COMPLETE の遷移を 3 回 polling で完了 (mock で 3 step)
- ERROR status で `KernelRunFailed` 例外
- CANCEL_ACKNOWLEDGED で expressive な戻り値

### `test_artifacts_output.py` (Step 7)
- `pull_kernel_output` が `kernels_output_cli(slug, path=tmp_dir)` を呼ぶ
- `place_into_run_dir` が `tmp_dir/runs/<run_id>/*` を `run_dir/*` にコピー
- `dvc_add` が `subprocess.run(["dvc", "add", str(run_dir)], cwd=repo_root, check=True)` を呼ぶ
- `dvc add` 失敗で `CalledProcessError` がそのまま伝播

### `test_artifacts_cost.py` (Step 7)
- 模擬 run.json (`kaggle_kernel_meta.runtime_seconds=1800`) 3 件で `total_gpu_hours_used=1.5`
- `kaggle_kernel_meta=None` の run.json は集計対象外
- 月跨ぎ filter: `month="2026-05"` で 2026-05-01〜2026-05-31 のみ
- markdown 出力に header / table / footer
- 出力先 path が `docs/experiment/kaggle_kernel_cost_report_2026-05.md`

## Integration Tests

### `test_cli_train.py` (Step 8)
End-to-end CLI flow with all external calls mocked:

```
python -m kaggle_kernel train abc1234 --case case1 --accelerator gpu-t4x2
  ↓
1. git rev-parse origin/abc1234 (subprocess mock → success)
2. load_kaggle_creds() (env mock → KaggleCreds)
3. dataset.api.push_dataset_version (KaggleApi mock → "v17")
4. kernel.template.render_notebook → /tmp/.../main.ipynb
5. dataset.metadata.write_dataset_metadata → /tmp/.../kernel-metadata.json
6. kernel.runner.push_kernel (KaggleApi mock → slug, version=3)
7. artifacts.launch.write_launch_json → run_dir/launch.json
```

Assertions:
- 全 step が正しい順序で呼ばれる
- run_dir に `launch.json` のみ存在 (artifact は pull 時に揃う)
- `kaggle_kernel_meta` の中身が `kernel_slug`, `kernel_version`, `dataset_slug`, `dataset_version`, `accelerator` を含む

### `test_cli_pull.py` (Step 8)
- `pull <run_id>` で `kernels_output_cli` → `place_into_run_dir` → `dvc add` の順で呼ばれる
- 既に run_dir が存在する場合は overwrite (or error)、policy 決定して test 化

## E2E Smoke (手動, Step 10)

実環境で 1 サイクル通す。費用 0 (無料枠)。

### 準備

```bash
# Kaggle API key を bot/.env に登録 (初回のみ)
echo "KAGGLE_USERNAME=<your-username>" >> bot/.env
echo "KAGGLE_KEY=<your-key>" >> bot/.env

# Rust wheel をローカルで pre-build
cd simulator/rust
maturin build --release --target x86_64-unknown-linux-gnu
# → simulator/rust/target/wheels/orbit_wars_rust-*.whl

# 初回 dataset push
dev/kaggle-kernel dataset push --commit-sha "$(git rev-parse HEAD)"
# → <user>/orbit-wars-bot dataset v1 が作成される
```

### サイクル実行

```bash
# 1) train 起動 + watch
dev/kaggle-kernel train "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2 --watch
# 期待:
#   [kaggle-kernel] dataset version: v2 (commit=abc1234)
#   [kaggle-kernel] kernel pushed: <user>/orbit-wars-case1-<run_id> v3
#   [kaggle-kernel] polling status... (60s interval)
#     - QUEUED (00:30)
#     - RUNNING (02:15)
#     - RUNNING (12:30)
#     - COMPLETE (32:42)
#   [kaggle-kernel] kernel complete. Run `dev/kaggle-kernel pull <run_id>` to fetch artifacts.

# 2) artifact pull
RUN_ID="$(ls -t data/output/models/imitation/case1/runs/ | head -1)"
dev/kaggle-kernel pull "$RUN_ID" --case case1
# 期待:
#   [kaggle-kernel] downloading output from <user>/orbit-wars-case1-...
#   [kaggle-kernel] placing into data/output/models/imitation/case1/runs/<run_id>/
#   [kaggle-kernel] dvc add data/output/models/imitation/case1/runs/<run_id>
#   [kaggle-kernel] run.json:
#     { "kaggle_kernel_meta": { ... } }

# 3) ローカル評価
ORBIT_WARS_WEIGHTS=data/output/models/imitation/case1/runs/$RUN_ID/best.pt \
  uv run --directory bot python -m pipeline.imitation.case1.evaluation.eval_vs_baseline \
  --episodes 30 --seed 0
# 期待: 既存 case1 と同等の出力形式、勝率 (smoke なので任意値で OK)

# 4) 採用 → promote (smoke では skip 可能)
dev/kaggle-kernel promote "$RUN_ID" --case case1
# 期待:
#   policy/weights.pt が更新される
#   run.json.status = "adopted"

# 5) cost-report
dev/kaggle-kernel cost-report --month "$(date +%Y-%m)"
# 期待:
#   docs/experiment/kaggle_kernel_cost_report_<YYYY-MM>.md が生成
#   total_gpu_hours_used: 0.55 (=33min)
```

### 成功基準

- [ ] kernel が COMPLETE で終了 (ERROR でない)
- [ ] `data/output/models/imitation/case1/runs/<run_id>/best.pt` が存在
- [ ] `run.json.kaggle_kernel_meta.kernel_slug` / `kernel_version` / `dataset_version` / `accelerator` が記録されている
- [ ] `dvc add` 後の `git status` で `.dvc` ファイルが untracked or modified
- [ ] `cost-report` が生成され、`total_gpu_hours_used` が 0 より大きい
- [ ] `dev/test-bot` グリーン

### 期待される失敗とリカバリ

| 失敗 | 原因仮説 | リカバリ |
|------|---------|---------|
| `kernel-metadata.json invalid` | `id` slug 命名不正 | `slug` を `<user>/orbit-wars-...` に修正 |
| `import orbit_wars_rust` ImportError | manylinux wheel mismatch | wheel rebuild + dataset version up |
| `pip install -e .` fail | `requires-python` mismatch | `pyproject.toml` を Kaggle Python に合わせ緩和 (Risk #11) |
| `kernels_output` 404 | kernel 未完了 / ERROR | `dev/kaggle-kernel logs <run_id>` で原因確認 |
| 9h timeout | training 長すぎ | `--max-hours 8.5` でセーフ終了 (将来実装) |
| KAGGLE_KEY 無効 | API key 期限切れ | `https://www.kaggle.com/settings` で再発行 |

## CI

- `dev/test-bot` (pytest + ruff + mypy) を全 Step で green 維持
- `bot/tests/src/kaggle_kernel/` は KaggleApi / subprocess を全 mock、CI で実 API は叩かない
- E2E smoke は **手動** で Step 10 でのみ実行、CI 化はしない (Kaggle API key を CI secret に置く運用コスト > 価値)

## Regression Surface

本基盤導入で他基盤に影響を与える可能性のある箇所:

1. **`bot/src/vast/run_meta.py`** — `RunMetadata` に field 追加 → 既存 vast/runpod の round-trip テストが全 green であること
2. **`bot/pipeline/imitation/case*/training/train.py`** — env 検出ブロック改修 → 既存 vast/runpod env 経路の test が全 green であること
3. **`bot/src/runpod_io/artifacts/run_meta.py`** — `promote_to_canonical` 共有 → 既存 RunPod promote test が green であること
4. **`bot/pyproject.toml`** — `kaggle>=1.6` 追加 → `uv sync` が成功すること、`runpod` / `vastai` SDK と coexist 可能なこと
