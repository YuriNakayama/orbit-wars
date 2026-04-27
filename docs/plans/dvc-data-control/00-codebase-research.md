# dvc-data-control — Codebase Research

本スコープは **ローカル環境のみ** を対象としたデータ管理基盤 (DVC + S3 remote)。
vast.ai 等のリモート GPU サーバ側設定は今回のスコープ外。

## Deep Codebase Analysis

### Area 1: データディレクトリ構造 (`data/`)

- **Files analyzed**:
  - `data/` (symlink → `/Users/user/project/orbit-wars/data/`)
  - `backend/src/dataset/storage/paths.py` (L1-L22)
  - `backend/src/dataset/storage/recorder.py`, `loader.py`, `analyze.py`
  - `backend/pipeline/imitation/case1/configs/il_baseline.yaml` (L4-L8)
- **Current implementation**:
  - ルートの `data/` は **worktree 横断で共有するため、メインリポジトリ `/Users/user/project/orbit-wars/data/` への symlink**。`.gitignore` では `data` は無視対象（`__pycache__/`・`.venv` 等と同様）で、git 管理されていない。
  - 3 層構造:
    - `data/lake/` (430 MB) — 原始データ。`kaggle_episodes/matches/` (356 MB / 1355 replays) と `selfplay/matches/` (21 MB / 45 replays)。配下で `index.parquet/` (hive-partitioned: `mode={mode}/run_{run_id}.parquet`) と `replays/{match_id}.json.gz` の 2 系統を持つ（`paths.py` で定義）。
    - `data/processed/` (0 B, 空) — 現状未使用。前処理中間成果物のスロット。
    - `data/mart/` (478 MB) — 学習用 parquet。`mart/imitation/case1/{train,val,train_q30,val_q30}.parquet` と `mart/case3/{train,val}.parquet`。
  - ルート直下のデバッグ CSV (`debug_splits_case4.csv` 等) と `submissions/` (344 KB) も現在同じ data/ に混在。
- **Key interfaces**:
  - `paths.index_root(data_root) = data_root / "matches" / "index.parquet"`
  - `paths.replays_root(data_root) = data_root / "matches" / "replays"`
  - recorder は `MatchRecord`（pydantic）と gzip JSON replay を書き、loader は `pl.scan_parquet(...)` + `gzip.decompress(...)` で読む。
  - 設定 YAML 側では **相対パス文字列** (`data/lake/...`, `data/mart/...`) を持ち、`backend/` を cwd とする前提で解決される。
- **Patterns used**:
  - Hive 分割 parquet + gzip JSON のハイブリッド。Polars lazy scan 中心。Pydantic schema (`MatchRecord`) による型境界。
  - パス解決は `data_root: Path` を引数注入する関数設計（グローバルシングルトンは無い）。
- **Coupling & side effects**:
  - `backend/src/dataset/selfplay/executor.py` と `backend/src/dataset/kaggle/scraper.py` が writer。
  - `backend/pipeline/imitation/case1/training/preprocess.py` が reader + mart writer。
  - 学習 (`training/train.py`) と評価 (`evaluation/*.py`) は mart parquet を直接 open する。
  - 重みファイル `backend/pipeline/imitation/case1/policy/weights.pt` は **git 管理下**（Kaggle 提出 tar.gz に含めるため）— DVC 化すると提出パスに影響。
- **Test coverage**:
  - `backend/tests/pipeline/imitation/case1/` 以下に determinism テスト (slow マーカー)。
  - dataset recorder/loader のユニットテストは snapshot 的な fixture で固定ルート。
- **Gaps identified**:
  - `data/` 全体が worktree 間で **共有 symlink**。複数 worktree が同時に writer を実行すると index.parquet の競合リスク。
  - チェックサム／バージョンの概念がなく、「誰がいつどのデータで学習したか」が手で追跡。
  - `data/processed/` は空スロットのまま（DVC 前処理中間ステージの置き場候補）。
  - Kaggle 提出物 (`policy/weights.pt`) と実験ごとのモデル候補の区別がない。

### Area 2: 設定ファイルとパイプライン実行経路

- **Files analyzed**:
  - `backend/pipeline/imitation/case1/configs/il_baseline.yaml`
  - `backend/pipeline/imitation/case1/training/preprocess.py` (L1-L60)
  - `backend/pipeline/imitation/case1/training/train.py` (L1-L40)
- **Current implementation**:
  - YAML に `data.kaggle_index_root`, `data.replay_dir`, `data.out_train`, `data.out_val`, `train.weights_out` をリテラル相対パスで記述。CLI は `typer` ベース、`--config path/to.yaml` で差し替え。
  - preprocess → train → eval は **人手で順番実行**。依存グラフは README に文章で記述のみ。
- **Key interfaces**:
  - `preprocess.py` が replay index を scan し、loser 側の `obs.step/player=None` を注入して parquet 2 分割 (train/val) を生成。
  - `train.py` が `CaseThreeDataset`（命名は旧）を読み BC 学習、`weights_out` に state_dict 保存。
- **Patterns used**:
  - YAML + typer CLI。モジュール境界に `dataclass(frozen=True)` の config。
  - 出力パスは YAML から受け取る設計で、再配置容易。
- **Coupling & side effects**:
  - `weights_out` は `pipeline/imitation/case1/policy/weights.pt` を直接指しており、学習完了と同時に **提出アーティファクトが上書き** される。
  - 入力 Kaggle replay が更新された場合、再前処理 → 再学習 → 再評価の「fan-out」は手動。
- **Gaps identified**:
  - 依存追跡（例: kaggle scrape を更新したら train も invalidate）の仕組みがない。
  - 評価結果（勝率、メトリクス）は `memory/project_*.md` と `docs/competition/*.md` に散在。構造化された評価レポートの版管理はない。

### Area 3: Git 管理範囲と既存の除外設定

- **Files analyzed**:
  - `.gitignore` (root)
  - `backend/pipeline/.submitignore`（Kaggle tar.gz 除外）
- **Current implementation**:
  - ルート `.gitignore` に `data` 行は **無く**、`data` は symlink そのものを git が無視していないように見えるが、実際は `/data` は root に存在しない（symlink は git のファイル判定上、symlink として追跡可能）。— 要確認。現状 `git ls-files` で `data` が出ないことを前提とする。
  - `backend/pipeline/.submitignore` は Kaggle 提出 tar.gz から `eda/ notebook/ evaluation/ training/ configs/` を除外。DVC メタファイル（`.dvc`）は現状無いので未考慮。
- **Gaps identified**:
  - DVC 運用開始時に `data/` を .gitignore で明示除外し、`*.dvc` / `dvc.yaml` / `dvc.lock` / `.dvcignore` を git 追跡する必要。
  - Kaggle 提出物 `weights.pt` の扱い（現状 git 管理）と DVC 管理の両立ポリシーが未定義。

### Area 4: インフラ（Terraform）

- **Files analyzed**: `infra/environment/dev/`, `infra/module/`, `.claude/rules/infra.md`
- **Current implementation**:
  - `infra/environment/dev/` のみ存在（空もしくは placeholder）。`module/` 配下は空。Terraform backend（S3 + DynamoDB state lock）は rules 内の推奨のみで実体無し。
  - 既存の AWS リソース（S3 bucket 等）は未デプロイ。
- **Gaps identified**:
  - DVC remote 用 S3 bucket、IAM ポリシー、（将来の）VPC エンドポイント等が未作成。
  - Terraform state backend 自体（state 用 S3 + lock table）もブートストラップが必要。

### Area 5: 既存の submissions / Kaggle アーティファクト経路

- **Files analyzed**: `backend/src/submit/` (ディレクトリ構造のみ), `backend/pipeline/imitation/case1/policy/`
- **Current implementation**:
  - `src/submit/archive / validator / uploader` で tar.gz ビルド → Kaggle upload。
  - `policy/weights.pt` は **サイズ < 1 MB** かつ「提出時点のスナップショット」という二つの性質を併せ持つ。git に残すメリット（Kaggle 提出直結）と DVC 化のメリット（実験ごとの候補保持）のトレードオフがある。
- **Gaps identified**:
  - 学習中間の候補重み（experiment checkpoint）は現状 `data/mart/...` にも `pipeline/.../policy/` にも置かれておらず、「試行ごとに手動リネーム」しているのが実態。

## Technical Constraints

- **Python 3.13 / uv** — DVC は Python ツールなので `uv add dvc[s3]` で導入可能。ただし DVC は pip 依存が複雑（boto3, gitpython, pygit2 等）、`uv.lock` の肥大に注意。
- **Worktree 共有の symlink** — `data/` がメインリポの symlink のため、**`.dvc` キャッシュも worktree で共有される**。複数 worktree で `dvc pull` / `dvc repro` を同時実行すると cache lock で衝突する可能性。
- **Kaggle 提出 tar.gz の制約** — packager は case_dir 配下の `.py/.json/.yaml/.pkl/.pt` を同梱。`.dvc` stub ファイルは拾われても Kaggle 実行環境から S3 は引けないため、**提出物に含める重みは git 管理（または tar.gz 直埋め）** が必須。
- **CI** — `dev/test-backend` は format/lint/mypy/pytest。DVC 関連ファイルは ruff/mypy 対象外（yaml）。`dvc.yaml` の構文検証は pre-commit / CI で別途。
- **S3 ネットワークコスト** — replay 356 MB + mart 478 MB + selfplay 21 MB ≈ **～850 MB**。初回 push/pull で S3 egress 課金 / 転送時間が発生。インクリメンタル運用が前提。

## Key Findings Summary

- **新設が必要なもの**:
  - `.dvc/config` と `.dvcignore`（リポジトリルート）
  - DVC パイプライン定義 `dvc.yaml`（stages: `scrape_kaggle` → `preprocess` → `train` → `eval`）と `params.yaml`（既存 YAML を DVC params 化 or ラップ）
  - S3 remote（`infra/module/application/dvc_remote/` か `infra/module/platform/storage/` に Terraform 化）
  - `.gitignore` の data 配下ルール更新
- **既存を活用できる点**:
  - 3 層データレイク構造（lake/processed/mart）は DVC stage 出力ディレクトリにそのまま対応可能。
  - YAML + typer CLI は DVC の `cmd:` フィールドから直接叩ける。
  - symlink による worktree 共有は DVC cache を `dvc cache dir` でメインリポと共有すれば `dvc pull` を 1 回で済ませられる利点に転化可能。
- **要トレードオフ判断**:
  - `weights.pt` を DVC 化するか git 管理のままにするか。
  - `data/lake/` 全体を DVC 管理下に置くか、`kaggle_episodes/` のように外部取得源のものはキャッシュとして `.dvcignore` するか。
  - Kaggle scraper / selfplay executor の **書き込み先**（DVC outs として宣言すると、実行後に `dvc commit` か `dvc repro` 内実行が必要になる）。
