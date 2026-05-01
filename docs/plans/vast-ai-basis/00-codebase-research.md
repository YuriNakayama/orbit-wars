# vast-ai-basis — Codebase Research

本スコープ: GitHub Actions を介さず、ローカル CLI から Vast.ai 上に GPU 学習用一時インスタンスを起動し、`dvc.yaml` の `train_imitation_case1` ステージを実行 → 成果物を DVC/S3 に push → ローカルで `dvc pull` して評価、というワークフロー基盤の設計と実装。Vast.ai は一時計算ノード扱い、正本は Git + DVC/S3。

## Deep Codebase Analysis

### Area 1: DVC パイプラインと params.yaml

- **Files analyzed**:
  - `dvc.yaml` (3 stages)
  - `params.yaml` (seed / data / model / train / inference / evaluation)
  - `dvc.lock` (現在の出力ハッシュ・サイズ)
  - `.dvc/config` (`remote = s3`, `url = s3://orbit-wars-dvc-286854171013/remote`, region=ap-northeast-1, sse=AES256)
  - `.dvcignore` (`.pytest_cache/`, `__pycache__/`, `*.pyc`, `.ruff_cache/`, `.mypy_cache/`)
- **Current implementation**:
  - 3 stage chain: `preprocess_imitation_case1` → `train_imitation_case1` → `eval_imitation_case1`。
  - `train_imitation_case1` の `outs` は **`backend/pipeline/imitation/case1/policy/weights.pt`** 1 個のみ（DVC 管理、git untracked）。
  - すべての stage cmd は `uv run --directory backend python -m pipeline.imitation.case1.training.<stage>`。
  - `preprocess` の入力 `data/lake/kaggle_episodes/matches` は約 370 MB / 1364 ファイル、出力 `train.parquet` (184 MB) + `val.parquet` (22 MB)。
  - `eval_imitation_case1` の `metrics`: `data/mart/imitation/case1/eval_metrics.json` (cache: false → git に直接コミット可)。
- **Key interfaces**:
  - `params.yaml` の `train.weights_out: backend/pipeline/imitation/case1/policy/weights.pt` がパスの正本。
  - `params.yaml` の `train.epochs=15`, `batch_size=256`, `lr=1.0e-3`, `num_workers=0`。Phase 2 で確定した `loss_weights` (focal α=0.75, γ=3.0, target_class_weight_beta=0.9999) は壊さない前提。
  - DVC remote は `s3` 1 個、AWS profile 名は `orbit-wars` (`dev/dvc-setup` で `--local` 設定)。
- **Patterns used**:
  - `dvc repro train_imitation_case1` で「変更があれば再実行」をハッシュ駆動で判定。
  - `dvc push` で `.dvc/cache` から `s3://orbit-wars-dvc-286854171013/remote` に同期。
  - 出力 `weights.pt` は git untracked / DVC 管理、`policy/weights_iter*.pt` は手動アーカイブ（git 管理）。
- **Coupling & side effects**:
  - `train_imitation_case1` を Vast.ai で実行する場合、Vast 側でも `.dvc/cache` を作って `dvc pull` → `dvc repro` → `dvc push` する必要あり。AWS credentials は DVC user (`orbit-wars-dev-dvc-user`) のキーを Vast に渡す必要がある。
  - 出力 `weights.pt` を上書きすると、ローカルでの Kaggle submit がそのまま影響を受ける（policy/agent.py が読むファイルそのもの）。
  - `params.yaml` を変更してから push しないと Vast 側の `dvc repro` は古い params で走る（Vast は `git checkout <sha>` を行うので、コミット粒度でしか同期しない）。
- **Test coverage**:
  - DVC 周りは `tests/` で直接の自動テストは無い。`backend/tests/pipeline/imitation/case1/` の slow テストが学習収束（CPU、tiny config）を検証。
- **Gaps identified**:
  - 学習 stage が **CPU 前提**: `train.py` は `device` 切替・`.to(device)` を一切持たない（`losses.py` 内に `device = from_multihot.device` の参照のみ）。GPU で動かすには `train()` 関数のデバイス対応が必要。
  - `weights.pt` 1 個しか stage 出力がない: 学習中の epoch metrics は `logger.info(json.dumps(...))` で stdout 出力のみ。`metrics.json` / `run.json` を別ファイルとして保存する仕組みは未実装。
  - 候補モデル管理が無い: 全実験が同じ `weights.pt` を上書きする。`weights_iter*.pt` は **手動コピーで運用** されており再現性が無い（git に commit してあるだけ）。

### Area 2: 学習コード (`backend/pipeline/imitation/case1/training/`)

- **Files analyzed**:
  - `train.py` (331 行)
  - `dataset.py` / `losses.py` / `preprocess.py`
- **Current implementation**:
  - `train()` は `cfg: dict` を受け取り、`DeepSetsPolicy` を CPU 上で学習し、最良 `val_total` の state_dict を `weights_out` に保存。`return TrainReport(epochs_run, best_val_loss, best_epoch, weights_path)`。
  - `_seed_all(seed)` は `torch.manual_seed` のみ（CUDA seed 設定なし）。
  - DataLoader は `num_workers=0`、`pin_memory` 未指定。
  - 各 epoch のメトリクスは `logger.info(json.dumps({"epoch":..., "train_total":..., ...}))` で stdout に出るだけ。集約ファイルは存在しない。
- **Key interfaces**:
  - `train(cfg)` → `TrainReport`。`cfg` は `params.yaml` 全体の dict。
  - `_load_params()` がリポジトリルート `params.yaml` を読む（`_repo_root()` 経由でハードコード）。
  - `app = typer.Typer(add_completion=False)` + `@app.command()` で `python -m pipeline.imitation.case1.training.train` を CLI 化。
- **Patterns used**:
  - 関数は純粋（cfg in / report out）。state は `train_loader` 経由でしか持たない。
  - `_abspath()` でリポジトリルート解決 → どこから呼んでも壊れない。
- **Coupling & side effects**:
  - 出力先は `params.yaml: train.weights_out` で完全に決まる。Vast 側で別パス `data/output/models/imitation/case1/runs/<run_id>/best.pt` に書く場合、`params.yaml` を上書きするか、`train()` に override パラメータを渡す形が必要。
  - `weights.pt` のパスは `policy/agent.py` の `_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"` でハードコード参照されており、Kaggle submit 時の正本になっている。
- **Test coverage**:
  - `backend/tests/pipeline/imitation/case1/training/` に collate / loss / determinism テスト。
  - GPU 経路のテストはなし。
- **Gaps identified**:
  - `train()` に `device` 引数がない。CUDA 利用には `model.to(device)`、batch tensor の `.to(device)` 注入、optimizer 後の `.cpu().state_dict()` 保存が必要。
  - `metrics.json` (epoch 履歴 + best val) と `run.json` (環境情報) を出力する hook が必要。`TrainReport` は最終 best のみで履歴を持たない。
  - run id 生成・保存ディレクトリ作成のロジックが無い（現状 `weights_out` の親ディレクトリのみ）。

### Area 3.5: submit DVC pull フック (補足調査)

- **Files analyzed**: `backend/src/submit/packager.py` (`ensure_dvc_artifacts()`), `backend/src/submit/__main__.py` (Step 0)
- **Current implementation**:
  - `submit_cmd` の **Step 0** で `ensure_dvc_artifacts(case_dir)` を必ず呼び、`dvc.yaml` の outs から case_dir 配下の欠損ファイルだけを抽出して `uv run --directory backend dvc pull <rels>` を実行。
  - 該当パスは現在 `pipeline/imitation/case1/policy/weights.pt` 1 個 (preprocess の出力 parquet は case_dir 外なので対象外)。
- **接続戦略**: Vast 学習で生成された候補 weights を「採用」するときは、まず `dvc pull` で **DVC 経由のローカル復元** を行い、その後 `dev/promote-weights` 等で `policy/weights.pt` にコピー → submit 時は ensure_dvc_artifacts が変更検知して整合性をチェックする、という運用が可能。
- **Gap**: candidate run dir (`data/output/models/imitation/case1/runs/<run_id>/best.pt`) は **submit パッケージに含まれない**（policy/ 配下のみ tar.gz に同梱）。Vast → submit 直結はできず、必ず昇格ステップを経由する設計が前提となる。

### Area 3: Submit パイプラインと weights.pt の正本性

- **Files analyzed**:
  - `backend/src/submit/packager.py` (推定: tar.gz パッケージング)
  - `backend/src/submit/validator.py`
  - `backend/pipeline/imitation/case1/main.py` / `policy/agent.py`
  - `backend/pipeline/.submitignore` (`eda/`, `notebook/`, `evaluation/`, `training/`, `configs/`)
- **Current implementation**:
  - Kaggle submit 時、`backend/pipeline/imitation/case1/` 全体が tar.gz に同梱される（`.submitignore` で `training/` と `evaluation/` は除外）。
  - `weights.pt` は同梱対象（policy/ 配下、`.submitignore` 対象外）。Kaggle 側でロードされるのはこの 1 ファイルだけ。
  - `dev/submit imitation/case1 -m "..."` が標準。`SubmissionStatus.ERROR` はクォータ消費しない（再挑戦可能）。
- **Key interfaces**:
  - `policy/agent.py` の `_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"` がただ 1 つの参照点。
  - submit 自動化 `dev/submit` が DVC pull のフックを持つ（`backend/src/submit/__main__.py` 周辺で確認可能）。
- **Patterns used**:
  - canonical model = `policy/weights.pt` という単一固定パス。
  - 候補モデルとの差は「採用したら policy/ にコピーする」運用。
- **Coupling & side effects**:
  - 任意の Vast 学習 run が直接 `policy/weights.pt` を上書きすると、**未検証モデルで Kaggle に submit してしまうリスク**。「Vast は候補ディレクトリ `data/output/models/.../runs/<run_id>/best.pt` に書き、採用後に手動で `policy/weights.pt` に昇格」という分離が必要。
- **Test coverage**:
  - 提出パッケージング / 相対 import / `Path.cwd()` の挙動は dry-run でカバー。weights.pt の差し替え経路自体はテスト無し。
- **Gaps identified**:
  - 「候補 weights を保存するディレクトリ」が DVC stage out に未定義。新しい stage out (例: `data/output/models/imitation/case1/runs/<run_id>/`) を `dvc.yaml` に追加すると、`dvc.yaml` の stage 構造を大きく変える必要がある（後述の検討事項）。
  - 採用フローを CLI 化する余地（例: `dev/promote-weights <run_id>` で run dir → policy/weights.pt をコピー）。

### Area 4: AWS / Terraform / DVC remote

- **Files analyzed**:
  - `infra/environment/dev/main.tf`, `variables.tf`, `outputs.tf`
  - `infra/module/application/dvc_remote/main.tf`
  - `dev/dvc-setup`
- **Current implementation**:
  - S3 bucket `orbit-wars-dvc-286854171013` (region ap-northeast-1, AES256, public access block 全有効、versioning enabled)。
  - IAM user `orbit-wars-dev-dvc-user` に `s3:ListBucket` と `s3:GetObject/PutObject` (prefix `remote/*`) のみ付与。アクセスキーは Terraform output (`dvc_iam_access_key_id` / `dvc_iam_secret_access_key`)。
  - `~/.aws/credentials` の profile `orbit-wars` 経由でローカル DVC が利用。
- **Key interfaces**:
  - DVC remote URL: `s3://orbit-wars-dvc-286854171013/remote`
  - IAM ポリシーは **`remote/` prefix 配下のオブジェクトの GetObject/PutObject のみ**。`s3:DeleteObject` は無いので破壊的操作不可（versioning ありで誤上書きも復旧可）。
- **Patterns used**:
  - `infra/module/application/<service>/` の再利用可能モジュール構造。state は最初ローカル、後で S3 backend に切替可能（`infra/environment/dev/README.md` 記載）。
- **Coupling & side effects**:
  - Vast 側でも `dvc pull` / `dvc push` するために、**同じ IAM user のクレデンシャルを Vast インスタンスに渡す**か、もしくは Vast 専用の IAM user を新規作成する必要がある。
  - 既存 IAM ポリシーには `s3:DeleteObject` が含まれない → DVC の garbage collection (`dvc gc -c`) は失敗する設計。Vast 側でも GC しない方針を維持できる。
- **Test coverage**: なし（インフラ）。
- **Gaps identified**:
  - Vast 専用 IAM user / アクセスキーを切るかどうか未決定。dev DVC user のキーを再利用する方が IAM resource を増やさず済むが、漏洩時の影響範囲が広がる。
  - Terraform のリソース追加余地: Vast クレデンシャル管理用に AWS Secrets Manager や SSM Parameter Store は **使わない**（ローカルから Vast へは直接 env で渡す方が単純）。

### Area 5: dev/ スクリプトと CLI 設計

- **Files analyzed**:
  - `dev/setup`, `dev/dvc-setup`, `dev/create-worktree`, `dev/submit`
- **Current implementation**:
  - すべて bash + set -euo pipefail。`cd "$(dirname "$0")/.."` でリポジトリルート移動。
  - `dev/submit` は `cd backend && exec uv run python -m submit submit "$@"` で Python CLI に委譲する thin wrapper。
  - `dev/create-worktree` は worktree 作成 + `data/` symlink + `.env` コピー。
- **Patterns used**:
  - **bash thin wrapper → Python CLI** の二段構成（`dev/submit` パターン）。複雑なロジックは Python 側 (`backend/src/submit/`) に集約。
  - 引数解析や複雑なフロー制御は Python (typer)。bash は `cd` と uv 起動だけ。
- **Coupling & side effects**:
  - すべて `backend/` で uv run することが前提。Vast 側でも同じ `cd backend && uv run` モデルを採用するのが自然。
- **Gaps identified**:
  - Vast.ai 用の thin wrapper `dev/vast-train` は未存在。新設が必要。
  - Vast.ai SDK (`vastai-sdk` または HTTP API + `requests`) を呼ぶ Python CLI も未存在。`backend/src/vast/` 等の新規パッケージを作る or `dev/vast-train` から直接 Python script を実行するかの判断が必要。

### Area 5.5: docs/experiment 既存運用との接続 (補足調査)

- **Files analyzed**: `docs/experiment/imitation_case1_iter9_summary.json`
- **Current schema** (代表): `iter`, `phase`, `description`, `training.{config,best_epoch,best_val_total,...}`, `val_metrics_vs_iter6`, `vs_baseline_v1_100ep`, `conclusion`, `decision`。
- **新 run.json への mapping**:
  - 既存 `training.config` (YAML path) → 新 `params_hash` (params.yaml の hash) と `git_sha`
  - 既存 `vs_baseline_v1_100ep` → 新フォーマットの `local_eval_results`（同等構造を保持）
  - 既存 `iter` フィールド → 新 `run_id` の suffix にエイリアスとして含める設計が可能（手動運用との混在期に有用）
- **方針**: 旧 iter6〜iter15 の summary はそのまま残置。新 Vast run は別命名空間 (`docs/experiment/<run_id>.json` または `data/output/.../runs/<run_id>/run.json` を正本) で管理し、iter 番号は付けない。

### Area 6: docs/experiment / docs/plans の運用

- **Files analyzed**:
  - `docs/experiment/` (既存 16 ファイル: iter6〜iter15 の summary / val_metrics / replays / threshold_sweep)
  - `docs/plans/dvc-data-control/` (00〜06 のテンプレ準拠サンプル)
- **Current implementation**:
  - 実験の結論は `docs/experiment/<date>_<topic>_result.md` (markdown) と `docs/experiment/imitation_case1_iter*.json` (構造化データ) の両建て。
  - 計画ドキュメントは `docs/plans/<feature-name>/00..06.md` の 7 分割。
- **Patterns used**:
  - 学習成果物は 1 iter ごとに `policy/weights_iter*.pt` (git 管理) として保存し、`docs/experiment/imitation_case1_iter*_summary.json` で対応付け。
- **Gaps identified**:
  - `weights_iter*.pt` を **git にコミット** している既存運用と、新方針「DVC 管理の `data/output/models/.../runs/<run_id>/best.pt`」がコンフリクト。移行方針の明示が必要（既存 iter9〜iter15 はそのまま git 残置、新規 run のみ DVC 経路）。

## Technical Constraints

1. **複数 worktree 同時実行は非推奨** (`.claude/CLAUDE.md` 記載): `.dvc/cache` を共有しているため `dvc pull/push` 同時実行は lock 競合のリスク。Vast 側との同時実行も同じ問題を踏む可能性 → ローカル評価は Vast の `dvc push` 完了後に明示的に実行。
2. **AWS IAM ポリシー上 DeleteObject 不可**: DVC GC は禁止。Vast 側でも `dvc gc -c` は実行しない方針が必須。
3. **Python 3.13 (`>=3.13,<3.14`)**: Vast インスタンスにこのバージョンの Python が用意できる image を選ぶ必要あり (`uv` インストール経由なら任意 base image でもよい)。
4. **`weights.pt` は Kaggle submit の正本**: 学習結果が直接ここに書かれると未検証モデルで submit する事故が起きる → run dir 経由で隔離する設計が必須。
5. **`train.py` は CPU 前提**: `model.to(device)` / batch `.to(device)` / `pin_memory` の追加が必要。`deterministic_algorithms(False)` の現状は維持してよい（DataLoader shuffle と CUDA 演算で完全 bit-exact 再現は諦める）。
6. **Kaggle submit 5 回/日の本番クォータ**: Vast 学習自体は本番 submit に直結しないが、評価結果を見て submit する場合は通常通り approval 必須。
7. **Vast.ai は使い捨てインスタンス**: SSH key、API key、stop/destroy の冪等性、課金タイマーをすべて考慮。`onstart` script の冪等化が信頼性のキー。

## Key Findings Summary

直接 feature 設計に効く findings：

- **canonical 単一パス問題**: `policy/weights.pt` は git untracked + DVC 管理 + Kaggle submit の正本という三役を兼ねている。Vast 学習の出力先は別ディレクトリ (`data/output/models/imitation/case1/runs/<run_id>/best.pt`) に分離し、採用時のみ昇格する方針が安全。
- **train.py の GPU 対応**: `device` 引数の追加と batch/model `.to(device)` の埋め込みが必須。CUDA seed 設定 (`torch.cuda.manual_seed_all`) と `pin_memory=True` も合わせて入れる。
- **DVC stage 拡張の選択肢**: 「`train_imitation_case1` の outs に run dir を追加する」 vs 「DVC 管理外の独立アーティファクト保存パスを作る」 の 2 択。前者は DVC のハッシュ駆動再実行と整合するが `dvc.yaml` 改修が大きい。後者は実装が薄いが run の lineage が DVC で追えない。要決定事項。
- **新規 stage `train_imitation_case1_run` を分岐**: 現状の `train_imitation_case1` (canonical weights.pt 出力) は維持し、Vast での候補生成は別 stage または直接 `python -m` で走らせ run dir に書く、という三本立ても候補。
- **Vast.ai SDK 選定**: `vastai-sdk` (公式 Python SDK) があるか、HTTP API + `requests` で手書きするか、CLI `vastai` バイナリをラップするかの 3 択（要 Web 調査）。
- **AWS credentials の Vast への渡し方**: env (`-e AWS_ACCESS_KEY_ID=...`) で渡すのが Vast.ai SDK 標準。漏洩リスクを抑えるため Vast 専用 IAM user を切る設計も検討余地あり。
- **既存 dev/ 構成との整合**: `dev/vast-train <commit-sha>` という bash thin wrapper + Python CLI (`backend/src/vast/`) という二段構成が既存パターンに合致。
- **冪等な onstart**: `git clone` は既に存在する場合 fetch のみ、`uv sync` は lockfile based で冪等、`dvc pull/repro/push` も DVC 側が再実行を判定するため冪等。実装上の鍵は **エラー時に AWS_* env を残したままインスタンスを落とさず触れる状態にするか、即停止するか** の選択。
