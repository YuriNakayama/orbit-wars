# dvc-data-control — Architecture Design

## 全体図

```
Local (macOS)                                        AWS ap-northeast-1
─────────────────────────────────────────────        ──────────────────────
Git worktrees (複数)                                  S3 bucket
  orbit-wars/                    ┐                    orbit-wars-dvc-<acct>/
  orbit-wars.worktrees/*/        │                      remote/
                                 │                        <md5 prefix>/<md5>
                                 │
     ↓ symlink / .dvc/config.local                      (versioning ON, SSE AES256)
                                 ↓
     メインリポ /Users/user/project/orbit-wars/
       .dvc/cache/               ←─── 共有 content-addressable cache
       data/                     ←─── symlink 先 (worktree 共有)
         lake/ mart/ processed/

                                 ↑ dvc push / pull (boto3 via aiobotocore)
                                 │
     Git: dvc.yaml / dvc.lock / params.yaml / .dvc/config / *.dvc
```

## ディレクトリ構成（追加・変更）

```
orbit-wars.worktrees/feature-dvc-data-control/
├── .dvc/
│   ├── config                 ← git 追跡。remote URL, cache type 等
│   ├── config.local           ← gitignore。cache.dir 絶対パス + aws profile
│   └── tmp/                   ← gitignore
├── .dvcignore                 ← git 追跡
├── .gitignore                 ← /data, /.dvc/tmp, /.dvc/config.local 等を追記
├── dvc.yaml                   ← git 追跡。stage 定義
├── dvc.lock                   ← git 追跡。自動生成ハッシュ
├── params.yaml                ← git 追跡。実験パラメータ（新規）
├── backend/
│   ├── pipeline/imitation/case1/
│   │   ├── configs/           ← 段階的に params.yaml へ移行 → 最終削除
│   │   └── policy/weights.pt  ← DVC 追跡に切替（git からは削除）
│   ├── src/submit/
│   │   └── packager.py        ← dvc pull 事前実行のフック追加
│   └── pyproject.toml         ← dvc[s3] 追加
├── infra/
│   ├── module/
│   │   ├── foundation/        ← (空のまま)
│   │   ├── platform/          ← (空のまま)
│   │   └── application/
│   │       └── dvc_remote/    ← 新規 module
│   │           ├── main.tf
│   │           ├── variables.tf
│   │           ├── outputs.tf
│   │           └── versions.tf
│   └── environment/
│       └── dev/
│           ├── main.tf        ← dvc_remote module 呼び出し
│           ├── variables.tf
│           ├── outputs.tf
│           ├── terraform.tfvars.example
│           └── versions.tf
├── dev/
│   ├── dvc-setup              ← 新規。dvc init + remote + cache dir 一括
│   └── submit                 ← 既存。内部で dvc pull を実行するよう更新
└── docs/plans/dvc-data-control/
    ├── 00-codebase-research.md
    ├── 01-web-research.md
    ├── 02-requirements.md
    ├── 03-architecture.md
    ├── 04-steps.md
    ├── 05-risks.md
    └── 06-testing.md
```

## Backend (Python) 設計

### `dvc.yaml` — Stage 定義

```yaml
stages:
  scrape_kaggle:
    cmd: uv run --directory backend python -m dataset.cli scrape-kaggle
    outs:
      - data/lake/kaggle_episodes/matches/index.parquet
      - data/lake/kaggle_episodes/matches/replays
    # scraper は外部状態依存 → 手動トリガー推奨。
    # デフォルトで always_changed=false にし、必要時のみ --force 実行。

  preprocess_imitation_case1:
    cmd: uv run --directory backend python -m pipeline.imitation.case1.training.preprocess
    deps:
      - data/lake/kaggle_episodes/matches
      - backend/pipeline/imitation/case1/training/preprocess.py
      - backend/pipeline/imitation/case1/policy/featurizer.py
      - backend/pipeline/imitation/case1/policy/geometry.py
      - backend/pipeline/imitation/case1/policy/templates.py
    params:
      - data
    outs:
      - data/mart/imitation/case1/train.parquet
      - data/mart/imitation/case1/val.parquet

  train_imitation_case1:
    cmd: uv run --directory backend python -m pipeline.imitation.case1.training.train
    deps:
      - data/mart/imitation/case1/train.parquet
      - data/mart/imitation/case1/val.parquet
      - backend/pipeline/imitation/case1/training/train.py
      - backend/pipeline/imitation/case1/training/losses.py
      - backend/pipeline/imitation/case1/training/dataset.py
      - backend/pipeline/imitation/case1/policy/model.py
    params:
      - model
      - train
      - seed
    outs:
      - backend/pipeline/imitation/case1/policy/weights.pt

  eval_imitation_case1:
    cmd: uv run --directory backend python -m pipeline.imitation.case1.evaluation.eval_vs_baseline --episodes 100 --seed 0
    deps:
      - backend/pipeline/imitation/case1/policy/weights.pt
      - backend/pipeline/imitation/case1/evaluation/eval_vs_baseline.py
    metrics:
      - data/mart/imitation/case1/eval_metrics.json:
          cache: false
```

**設計上のポイント**:

- **category/case ごとに命名** (例 `preprocess_imitation_case1`)。将来 `case2`, `rulebase/*` 追加時に平行展開可能。
- `cmd` は `uv run --directory backend` で統一し、`backend/` を cwd にする（pipeline.md ルール準拠）。
- `deps` は **Python ソースファイルも含める**。コード変更で再実行されるように（コメント変更は md5 差分検出）。
- `metrics` は `cache: false` で git 直追跡（勝率 JSON は小さく、履歴を git で見たい）。
- `scrape_kaggle` は `deps` なしで運用。通常は手動 `dvc repro -f scrape_kaggle`。
- `weights.pt` を outs 宣言すると DVC が自動で `.gitignore` に追加 → git 管理から外れる。Kaggle 提出フロー側で pull が必要。

### `params.yaml` — パラメータファイル（新規）

```yaml
seed: 0

data:
  kaggle_index_root: data/lake/kaggle_episodes/matches/index.parquet
  replay_dir: data/lake/kaggle_episodes/matches/replays
  out_train: data/mart/imitation/case1/train.parquet
  out_val: data/mart/imitation/case1/val.parquet
  rating_quantile: 0.50
  val_split: 0.10
  modes: ["1v1"]
  max_episodes: null

model:
  planet_in_dim: 11
  global_in_dim: 6
  hidden: 128
  ships_buckets: 4

train:
  batch_size: 256
  epochs: 15
  lr: 1.0e-3
  weight_decay: 1.0e-4
  num_workers: 0
  loss_weights:
    from: 1.0
    target: 2.0
    ships: 0.5
    from_pos_weight: 8.5
    from_focal_gamma: 2.0
    from_focal_alpha: 0.25
    target_label_smoothing: 0.1
    target_entropy_bonus: 0.05
  weights_out: backend/pipeline/imitation/case1/policy/weights.pt

inference:
  from_threshold: 0.05
```

- 既存 `configs/il_baseline.yaml` と構造は揃える（移行コスト最小化）。
- `preprocess.py` / `train.py` は **`--config path/to.yaml` 引数廃止 → `params.yaml` 固定読み込み** に変更（検証プロジェクト方針で後方互換不要）。
- DVC 的には `params: [data]` / `params: [model, train, seed]` でトップレベルキー単位の granular 追跡が有効になる。

### `backend/src/submit/packager.py` — 提出前の `dvc pull` フック

```python
# 既存: tar.gz ビルド前のファイル収集
# 追加: weights.pt が DVC-tracked なら事前に dvc pull
def ensure_weights(case_dir: Path) -> None:
    weights = case_dir / "policy" / "weights.pt"
    if not weights.exists() or _is_dvc_stub(weights):
        subprocess.run(
            ["dvc", "pull", str(weights)],
            cwd=_repo_root(), check=True,
        )
```

- `dev/submit` からの呼び出し経路で自動化するため、ユーザーは `dvc pull` を明示実行する必要がない。
- `--skip-dvc-pull` フラグは最小構成では設けない（必要になったら追加）。

### `backend/pyproject.toml` — 依存追加

```toml
dependencies = [
    # ... 既存 ...
    "dvc[s3]>=3.55",
]
```

- `dev` group には追加しない（Kaggle 提出環境では実行時に使わない）。ただし `uv sync` で常に入るため、Kaggle tar.gz には同梱されない（pipeline 側ロジックには import されない）。

## Data Model (S3 Layout)

```
s3://orbit-wars-dvc-<account_id>/
├── remote/                         ← DVC default remote (content-addressable)
│   ├── ab/
│   │   └── cdef0123...             ← md5 の先頭 2 文字で prefix 分割
│   └── ...
└── (他 prefix は将来用、今回は remote/ のみ)
```

- DVC 3.x の content-addressable storage 形式 (`<prefix>/<hash>`) に従う。
- bucket 単独で他用途と共有しないポリシー（prefix で切り分けすると IAM がややこしい）。

## Infrastructure (Terraform)

### `infra/module/application/dvc_remote/main.tf`

```hcl
resource "aws_s3_bucket" "dvc_remote" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_versioning" "dvc_remote" {
  bucket = aws_s3_bucket.dvc_remote.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_remote" {
  bucket = aws_s3_bucket.dvc_remote.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_remote" {
  bucket                  = aws_s3_bucket.dvc_remote.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_user" "dvc_user" {
  name = "${var.prefix}-dvc-user"
}

resource "aws_iam_access_key" "dvc_user" {
  user = aws_iam_user.dvc_user.name
}

resource "aws_iam_user_policy" "dvc_remote_rw" {
  user = aws_iam_user.dvc_user.name
  name = "${var.prefix}-dvc-remote-rw"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid = "ListBucket"
        Effect = "Allow"
        Action = ["s3:ListBucket"]
        Resource = aws_s3_bucket.dvc_remote.arn
      },
      {
        Sid = "ReadWriteObjects"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.dvc_remote.arn}/remote/*"
      }
      # DeleteObject は付与しない（dvc gc -c は管理者ロール別途）
    ]
  })
}
```

### `infra/environment/dev/main.tf`

```hcl
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  # 初回は local state で bootstrap。state bucket 作成後に s3 backend へ移行。
}

provider "aws" {
  region = "ap-northeast-1"
  default_tags {
    tags = {
      Project     = "OrbitWars"
      Environment = "dev"
      ManagedBy   = "Terraform"
    }
  }
}

module "dvc_remote" {
  source      = "../../module/application/dvc_remote"
  bucket_name = var.dvc_bucket_name
  prefix      = "orbit-wars-dev"
}
```

- `terraform.tfvars.example` に `dvc_bucket_name = "orbit-wars-dvc-<ACCOUNT_ID>"` のテンプレ。実 tfvars は gitignore。
- 今回は **plan まで通す**。apply はユーザー承認後に実行。

## Cache & Config Layout

### `.dvc/config` (git 追跡)

```ini
[core]
    remote = s3
    autostage = true
['remote "s3"']
    url = s3://orbit-wars-dvc-<REPLACE_WITH_ACCOUNT>/remote
    region = ap-northeast-1
    sse = AES256
[cache]
    type = symlink
    shared = group
```

### `.dvc/config.local` (gitignore、各開発者で個別)

```ini
[cache]
    dir = /Users/user/project/orbit-wars/.dvc/cache
['remote "s3"']
    profile = orbit-wars
```

### `.dvcignore` (git 追跡)

```
# DVC がスキャンを省略するパス（case の `.submitignore` や tmp）
.pytest_cache/
__pycache__/
*.pyc
```

### `.gitignore` 追記分

```
# DVC
/.dvc/tmp
/.dvc/config.local
/.dvc/cache
# Data (DVC-managed)
/data
# Weights (DVC-managed, was git-tracked before)
backend/pipeline/imitation/case1/policy/weights.pt
# Terraform
infra/**/.terraform/
infra/**/*.tfstate
infra/**/*.tfstate.backup
infra/**/terraform.tfvars
```

## External Integrations

- **AWS S3** (ap-northeast-1): DVC remote。boto3 経由でアクセス。
- **AWS IAM**: 専用 user + access key。AWS CLI profile `orbit-wars` でローカル利用。

## 将来拡張（メモのみ、今回実装しない）

- **GPU サーバ連携**: vast.ai / EC2 で同じ DVC remote を pull。IAM role は別途（instance profile 推奨）。
- **CI 連携**: `.github/workflows/cd-kaggle-submit.yml` から OIDC で AWS 認証 → `dvc pull` → tar.gz → Kaggle upload。
- **複数 remote**: backup remote (S3 別 region) を `dvc remote add --default=false backup s3://...` で追加。
- **DVC Experiments**: `dvc exp run` / `dvc exp push` で複数 run の比較。
