# vast-ai-basis — Implementation Steps

実装は **train.py の GPU 対応 → vast パッケージ → onstart → CLI/promote/cost** の順で進める。各ステップで unit test を同時にコミットし、`dev/test-backend` がグリーンな状態を維持する。Step 内の **並列可能** マークは独立に着手可能なペアを示す。

---

## Step 1: data/output/ ディレクトリと .gitignore セットアップ

**Target**: cross-cutting
**Dependencies**: None

### Overview
新規 run dir 群を git ignore かつ DVC で個別管理できるように、`data/output/` を gitignore に追加し、その下のディレクトリ構造の README を置く。

### Work Items
- [ ] `.gitignore` に `data/output/` を追加
- [ ] `data/output/models/imitation/case1/runs/.gitkeep` で空 dir を git 管理（`.dvc` ファイルだけは後で commit する）
- [ ] `data/output/README.md` で「ここは vast ai 学習成果物の置き場、git untracked、DVC 管理」とドキュメント化

### Target Files
- `.gitignore`
- `data/output/README.md`
- `data/output/models/imitation/case1/runs/.gitkeep`

### Acceptance Criteria
- `git status` で `data/output/models/imitation/case1/runs/<some_id>/` に書き込んでも tracked にならない
- `data/output/README.md` に運用方針が書かれている

---

## Step 2: train.py の GPU 対応 + run dir override (並列可能 with Step 3)

**Target**: backend
**Dependencies**: Step 1

### Overview
`train.py` を CPU/GPU 両対応にし、環境変数 `ORBIT_WARS_RUN_DIR` が指定されていれば そのディレクトリに `best.pt` / `metrics.json` / `run.json` を出力する。

### Work Items
- [ ] `_seed_all` に `torch.cuda.manual_seed_all(seed)` を追加
- [ ] `train()` 内で `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- [ ] `model.to(device)` と DataLoader に `pin_memory=(device.type=="cuda")`
- [ ] `_to_batch_features` の後に `.to(device, non_blocking=True)` を batch tensor 全部に chain
- [ ] `train()` の戻り値 `TrainReport` に `train_loss_history`, `val_loss_history`, `device`, `runtime_seconds` を追加
- [ ] `weights_out` を env override に対応させる（`ORBIT_WARS_RUN_DIR` 優先）
- [ ] env が指定されているとき、`metrics.json` を run dir に出力（cache: false 互換）
- [ ] env が指定されているとき、`run.json` を run dir に出力（vast.run_meta は次 Step で導入するため、本 Step では一旦 minimal な dict→json で実装、後で run_meta.py に置換するメモを TODO コメントで記録）
- [ ] `state_dict` を保存するときは `model.cpu().state_dict()` で CPU テンソル化（再ロード時の互換性）
- [ ] `tests/pipeline/imitation/case1/training/test_train_run_dir.py` を新設し、env override 時に run dir に 3 ファイル生成されることを検証 (CPU only でテスト)

### Target Files
- `backend/pipeline/imitation/case1/training/train.py`
- `backend/tests/pipeline/imitation/case1/training/test_train_run_dir.py`

### Acceptance Criteria
- 既存テスト群 (`pytest backend/tests/pipeline/imitation/case1`) が全部 pass
- 新テストで `ORBIT_WARS_RUN_DIR=/tmp/run_xxx` を設定すると `/tmp/run_xxx/{best.pt, metrics.json, run.json}` の 3 ファイルが生成されることを確認
- `dvc repro train_imitation_case1` を実行すると、env なしで従来通り `policy/weights.pt` のみ更新（regression 確認）

---

## Step 3: backend/src/vast/run_meta.py + tests (並列可能 with Step 2)

**Target**: backend
**Dependencies**: None

### Overview
RunMetadata dataclass、run_id 生成、params hash、JSON I/O を実装する。Vast の依存はゼロなので並列着手可能。

### Work Items
- [ ] `backend/src/vast/__init__.py` (空)
- [ ] `backend/src/vast/run_meta.py`:
  - `RunMetadata` dataclass (schema_version=1 を含める)
  - `generate_run_id(branch, sha, seed, *, now=None) -> str` — slug 生成 (英数 + `_-` のみ、branch 中の `/` は `-` に変換)
  - `hash_params(params_yaml_path) -> str` — yaml.safe_load → json.dumps(sort_keys=True) → sha256 → 先頭 12 文字
  - `write_run_json(run_dir, meta) -> None` — atomic write (tmp file → rename)
  - `update_run_json(run_dir, **patch) -> RunMetadata` — read-modify-write、`updated_at` を必ず更新
- [ ] `backend/tests/vast/test_run_meta.py`:
  - run_id format の正規表現マッチ
  - branch slug の `/` -> `-` 変換
  - params_hash が同じ params.yaml で deterministic (順序非依存)
  - write/update の round-trip

### Target Files
- `backend/src/vast/__init__.py`
- `backend/src/vast/run_meta.py`
- `backend/tests/vast/__init__.py`
- `backend/tests/vast/test_run_meta.py`

### Acceptance Criteria
- pytest backend/tests/vast/test_run_meta.py が pass
- generate_run_id は決定論 (now を引数で渡せる)
- params_hash は dict 内のキー順序を変えても同じ値を返す

---

## Step 4: train.py を vast.run_meta に統合

**Target**: backend
**Dependencies**: Step 2, Step 3

### Overview
Step 2 で minimal dict として書いていた run.json を `RunMetadata` に置き換える。

### Work Items
- [ ] `train.py` の run.json 出力部を `from vast.run_meta import RunMetadata, write_run_json` に置換
- [ ] env から `git_sha` (`ORBIT_WARS_GIT_SHA`), `git_branch` (`ORBIT_WARS_GIT_BRANCH`), `vast_instance_id` (`ORBIT_WARS_VAST_INSTANCE_ID`), `gpu_name` (`torch.cuda.get_device_name(0)`) を取得
- [ ] env が無い場合（ローカル run）は git 直接呼び出しで sha/branch を解決（`subprocess.run(["git", "rev-parse", "HEAD"])`）
- [ ] `command` フィールドは `os.environ.get("ORBIT_WARS_COMMAND", "manual")` から取得（onstart で `ORBIT_WARS_COMMAND` を設定する）
- [ ] `params_hash` は `hash_params(_repo_root() / "params.yaml")` で計算
- [ ] `tests/pipeline/imitation/case1/training/test_train_run_dir.py` を更新し、run.json の schema_version=1 と必須フィールドを assert

### Target Files
- `backend/pipeline/imitation/case1/training/train.py`
- `backend/tests/pipeline/imitation/case1/training/test_train_run_dir.py`

### Acceptance Criteria
- run.json に schema_version, run_id, git_sha, git_branch, params_hash, seed, command, weights_path, train_metrics, status="pushed", created_at, updated_at が揃う
- env 未設定でも fallback で動く（ローカルでの sanity 確認）

---

## Step 5: backend/src/vast/auth.py

**Target**: backend
**Dependencies**: None

### Overview
AWS credentials と VAST_API_KEY を読み込む補助モジュール。

### Work Items
- [ ] `auth.load_aws_creds(profile: str = "orbit-wars") -> AwsCreds` — `subprocess.run(["aws", "configure", "get", "aws_access_key_id", "--profile", profile])` で取得
- [ ] `auth.load_vast_api_key() -> str` — `python-dotenv` で `backend/.env` を読み、`VAST_API_KEY` を返す。無ければ環境変数フォールバック、それも無ければ actionable error
- [ ] `backend/.env.example` に `VAST_API_KEY=` の行追加
- [ ] `backend/tests/vast/test_auth.py` — env path のテスト (mock subprocess)

### Target Files
- `backend/src/vast/auth.py`
- `backend/.env.example`
- `backend/tests/vast/test_auth.py`

### Acceptance Criteria
- `aws configure get` が呼ばれるが、テストでは monkeypatch で固定値を返す
- VAST_API_KEY 未設定時は `RuntimeError` で actionable message

---

## Step 6: backend/src/vast/offers.py + tests (並列可能 with Step 7)

**Target**: backend
**Dependencies**: Step 5

### Overview
vastai SDK の search_offers をラップ、rich Table 表示、対話的 pick。

### Work Items
- [ ] `pyproject.toml` の dependencies に `vastai>=0.3.0` 追加 (or 最新安定 minor)
- [ ] `pyproject.toml` の `[tool.hatch.build.targets.wheel] packages` に `src/vast` を追加
- [ ] `Offer` dataclass + `search_offers()` 実装。SDK `VastAI().search_offers(query=...)` を呼び、応答 dict を Offer に変換
- [ ] `format_table(offers)` で `rich.Table` を返す（# / GPU / num / dph / reliability / cuda / region）
- [ ] `pick_offer(offers)` で `rich.prompt.IntPrompt.ask` で番号入力 → 範囲外なら再入力
- [ ] `tests/vast/test_offers.py` — `VastAI` を mock、固定 dict を返して Offer 変換と sort（dph asc）を確認

### Target Files
- `backend/pyproject.toml`
- `backend/src/vast/offers.py`
- `backend/tests/vast/test_offers.py`

### Acceptance Criteria
- pytest pass
- `uv sync` で vastai がインストールされる

---

## Step 7: backend/src/vast/instance.py + onstart.sh.tmpl (並列可能 with Step 6)

**Target**: backend
**Dependencies**: Step 5

### Overview
onstart テンプレートの sed 置換と `create_instance` ラッパ。

### Work Items
- [ ] `backend/src/vast/onstart.sh.tmpl` をアーキ設計通りに作成（`<COMMIT_SHA>`, `<RUN_ID>`, `<STAGE>`, `<BRANCH>`, `<REPO_URL>` 5 箇所）
- [ ] `instance.render_onstart(template_path, **vars) -> Path` — `tempfile.NamedTemporaryFile(delete=False, suffix=".sh")` に書き、各 placeholder を `str.replace` で置換
- [ ] sanitize: vars の値を `^[A-Za-z0-9._/:\\-]+$` で正規表現バリデーション、不正文字でエラー (shell injection 防止)
- [ ] `instance.build_env_string(...) -> str` — `'-e KEY1=VAL1 -e KEY2=VAL2 ...'` 形式の文字列。各値を shlex.quote
- [ ] `instance.create_instance(offer_id, ...)` — VastAI SDK 呼び出し、戻り値の instance id を int で返す
- [ ] `tests/vast/test_instance.py` — render_onstart の置換をテンプレ全行で検証、env_string の形式を assert、shell injection 試行 (e.g. `; rm -rf /`) で例外
- [ ] `tests/vast/test_onstart_template.py` — テンプレ自体を bash -n でパース可能（syntax check）

### Target Files
- `backend/src/vast/instance.py`
- `backend/src/vast/onstart.sh.tmpl`
- `backend/tests/vast/test_instance.py`
- `backend/tests/vast/test_onstart_template.py`

### Acceptance Criteria
- pytest pass
- `bash -n backend/src/vast/onstart.sh.tmpl` (placeholder 含み) が syntax error にならない
- shell injection の attempt は ValueError でブロック

---

## Step 8: backend/src/vast/cli.py + python -m vast entry

**Target**: backend
**Dependencies**: Step 4, Step 5, Step 6, Step 7

### Overview
4 サブコマンド (`train`, `pull`, `promote`, `cost-report`) を統合した typer CLI。

### Work Items
- [ ] `backend/src/vast/__main__.py`: `from vast.cli import app; app()`
- [ ] `backend/src/vast/cli.py`:
  - `train` サブコマンド (実装中身は Step 9)
  - `pull` サブコマンド (実装は Step 10)
  - `promote` サブコマンド (実装は Step 11)
  - `cost-report` サブコマンド (実装は Step 12)
  - 各サブコマンドは最初は stub で OK
- [ ] `tests/vast/test_cli.py` — typer CliRunner でサブコマンドが registered されていることを smoke test

### Target Files
- `backend/src/vast/__main__.py`
- `backend/src/vast/cli.py`
- `backend/tests/vast/test_cli.py`

### Acceptance Criteria
- `cd backend && uv run python -m vast --help` が 4 つのサブコマンドを表示
- pytest pass

---

## Step 9: cli.py train 実装 + dev/vast-train

**Target**: backend, dev tooling
**Dependencies**: Step 8

### Overview
`vast train <sha>` で git 検証 → search → pick → render → create_instance を実行。

### Work Items
- [ ] git 検証ヘルパ: `git cat-file -e <sha>` で存在確認、`git branch -r --contains <sha>` で push 確認
- [ ] commit-sha が unpushed の場合は `typer.BadParameter` で fail
- [ ] dirty working tree の警告（commit-sha は変わらないが UX）
- [ ] cost 推定: `dph_total * 0.5h`、`--cost-limit` 超過時 `typer.confirm` (デフォルト No)
- [ ] `dev/vast-train` (bash thin wrapper): `cd backend && exec uv run python -m vast train "$@"`
- [ ] chmod +x dev/vast-train
- [ ] `tests/vast/test_cli_train.py` — CliRunner + mock (offers, instance, auth)、git 検証は subprocess を monkeypatch

### Target Files
- `backend/src/vast/cli.py`
- `dev/vast-train`
- `backend/tests/vast/test_cli_train.py`

### Acceptance Criteria
- `dev/vast-train abc1234 --stage train_imitation_case1` (mock 環境下) で正常 path がテスト pass
- unpushed sha で fail-fast を確認
- pytest pass

---

## Step 10: cli.py pull 実装 + dev/vast-pull

**Target**: backend, dev tooling
**Dependencies**: Step 8

### Overview
DVC pull で run dir をローカルに取得。

### Work Items
- [ ] `vast pull <run_id>`: `subprocess.run(["uv", "run", "--directory", "backend", "dvc", "pull", f"data/output/models/imitation/case1/runs/{run_id}"])` を実行
- [ ] pull 後、`run.json` を rich で pretty-print
- [ ] `status` が `pushed` 以外（`failed`/`running` 等）なら警告ログ
- [ ] `dev/vast-pull` thin wrapper
- [ ] `tests/vast/test_cli_pull.py` — subprocess を mock

### Target Files
- `backend/src/vast/cli.py`
- `dev/vast-pull`
- `backend/tests/vast/test_cli_pull.py`

### Acceptance Criteria
- pull コマンドが正しい dvc pull 引数を組み立てる
- pytest pass

---

## Step 11: cli.py promote 実装 + dev/vast-promote

**Target**: backend, dev tooling
**Dependencies**: Step 10

### Overview
candidate weights を canonical に昇格。

### Work Items
- [ ] `vast promote <run_id>`:
  - `<run_dir>/best.pt` の存在確認
  - `cp <run_dir>/best.pt backend/pipeline/imitation/case1/policy/weights.pt`
  - `subprocess.run(["uv", "run", "--directory", "backend", "dvc", "commit", "pipeline/imitation/case1/policy/weights.pt"])`
  - `update_run_json(run_dir, status="adopted", local_eval_results=...)` (オプション: --eval-results path で json 取り込み)
  - `subprocess.run(["uv", "run", "--directory", "backend", "dvc", "add", run_dir])` で再 push 準備
  - 最後に `git status` を表示し、ユーザに git commit を促す（自動 commit はしない）
- [ ] `dev/vast-promote` thin wrapper
- [ ] `tests/vast/test_cli_promote.py`

### Target Files
- `backend/src/vast/cli.py`
- `dev/vast-promote`
- `backend/tests/vast/test_cli_promote.py`

### Acceptance Criteria
- weights.pt の cp が行われる
- run.json の status が adopted に更新される
- pytest pass

---

## Step 12: cost.py + cli.py cost-report + dev/vast-cost-report

**Target**: backend, dev tooling
**Dependencies**: Step 8

### Overview
月次コスト集計レポート。

### Work Items
- [ ] `backend/src/vast/cost.py`:
  - `aggregate_runs(runs_root: Path, month: str | None) -> CostReport`
  - `render_markdown(report: CostReport) -> str`
- [ ] `vast cost-report` サブコマンド: aggregate → markdown → `docs/experiment/vast_cost_report_<YYYY-MM>.md` に保存
- [ ] `dev/vast-cost-report` thin wrapper
- [ ] `tests/vast/test_cost.py` — fixtures に小さい run.json 群を置き、月単位集計が正しい合計値を返すことを assert

### Target Files
- `backend/src/vast/cost.py`
- `backend/src/vast/cli.py`
- `dev/vast-cost-report`
- `backend/tests/vast/test_cost.py`

### Acceptance Criteria
- 3 つの run.json で集計するとそれぞれ合計と平均が正しい
- markdown 出力が `docs/experiment/` に書かれる
- pytest pass

---

## Step 13: 全体 e2e dry-run + ドキュメント

**Target**: cross-cutting
**Dependencies**: Step 9, 10, 11, 12

### Overview
本物の Vast 起動はせずに、CLI の wiring と `dev/test-backend` (format/lint/type/pytest) のグリーンを保つ。実環境テストはユーザー判断で行う。

### Work Items
- [ ] `dev/test-backend` を実行してグリーン確認
- [ ] `docs/plans/vast-ai-basis/README.md` (オプション) で運用手順を 1 ページに集約
- [ ] `backend/pipeline/imitation/case1/README.md` に新フローへの参照リンクを追加
- [ ] `.claude/CLAUDE.md` の Commands セクションに `dev/vast-train` 等を追記
- [ ] mypy, ruff の対応（typer のオプション型は `Annotated[...]` 推奨）

### Target Files
- `backend/pipeline/imitation/case1/README.md`
- `.claude/CLAUDE.md`
- `docs/plans/vast-ai-basis/README.md` (任意)

### Acceptance Criteria
- `dev/test-backend` がグリーン
- README が新フローを参照している

---

## Cross-Step Dependencies

```
Step 1 (data/output/)
  └─> Step 2 (train.py GPU/run-dir)  ────┐
                                         ▼
Step 3 (run_meta.py) ──────────────────> Step 4 (train.py + run_meta integration)

Step 5 (auth.py) ──┬─> Step 6 (offers.py)  ──┐
                   └─> Step 7 (instance.py + onstart.sh.tmpl) ─┐
                                                                ▼
                                                           Step 8 (cli.py stub)
                                                                ▼
                                            ┌──────────┬───────┴───────┬──────────┐
                                            ▼          ▼               ▼          ▼
                                       Step 9     Step 10        Step 11     Step 12
                                       (train)    (pull)         (promote)   (cost-report)
                                            └──────────┴───────────────┴──────────┘
                                                                ▼
                                                           Step 13 (e2e + docs)
```

並列推奨ペア: (Step 2, Step 3)、(Step 6, Step 7)、(Step 9 〜 Step 12 の 4 つは Step 8 完了後に並列可能).
