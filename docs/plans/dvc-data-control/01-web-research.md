# dvc-data-control — Web Technical Research

## 1. Official DVC Documentation

### 1a. パイプライン定義 (`dvc.yaml` / `params.yaml` / `dvc.lock`)

- [Get Started: Data Pipelines](https://doc.dvc.org/start/data-pipelines/data-pipelines)
- [dvc.yaml Files](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files)

**主要フィールド**（本プロジェクトで利用するもの）:

| フィールド | 用途 |
|---|---|
| `cmd` | 実行シェルコマンド（`uv run python -m ...` 等） |
| `deps` | 入力依存（ファイル/ディレクトリ）。変更検知で stage 再実行 |
| `outs` | DVC 追跡出力。自動で `.gitignore` に追加され remote push 対象 |
| `params` | `params.yaml` のキーを granular に追跡（値変更で再実行） |
| `metrics` | 評価指標の小さな JSON/YAML。git 追跡推奨 |
| `wdir` | cmd 実行時の cwd（本プロジェクトは `backend/` を想定） |
| `frozen` | true にすると dvc repro でスキップ（外部由来データの保護） |
| `always_changed` | Kaggle scraper のように外部状態依存を毎回実行したい場合 |

**foreach / matrix**: `cases` ごとに繰り返す stage 定義が可能（将来 imitation/case2, case3... に拡張）。

**依存解決モデル**: `dvc.lock` に各 stage の deps/outs の md5 を記録し、`dvc repro` 時にハッシュ差分で最小再実行。Git で `dvc.yaml` + `dvc.lock` を commit すれば「コードと一緒にパイプライン履歴も再現可能」。

### 1b. S3 Remote 設定

- [Amazon S3 Remote](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3)
- [`dvc remote modify`](https://dvc.org/doc/command-reference/remote/modify)

**設定コマンドシーケンス**:

```bash
# git 追跡する共通設定（チーム/環境横断）
dvc remote add -d s3 s3://orbit-wars-dvc/remote
dvc remote modify s3 region ap-northeast-1
dvc remote modify s3 sse AES256            # SSE 強制

# ローカル限定（.dvc/config.local、git ignored）
dvc remote modify --local s3 profile orbit-wars    # AWS CLI profile
```

**認証オプション優先順位**:
1. 環境変数 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
2. `profile`（AWS CLI profile、`~/.aws/credentials`）
3. `access_key_id` / `secret_access_key` を `config.local` に直書き（非推奨）

**本プロジェクト推奨**: AWS CLI profile `orbit-wars` を使い `config.local` に `profile` だけを記述。secret は `~/.aws/credentials` に保管し git に入らない。

### 1c. 共有 Cache 設定

- [How to Share a DVC Cache](https://doc.dvc.org/user-guide/how-to/share-a-dvc-cache)

**設定**（本プロジェクト向け）:

```bash
# メインリポ側の 1 箇所に集約
dvc cache dir /Users/user/project/orbit-wars/.dvc/cache    # 絶対パス
dvc config cache.type symlink                               # copy でなく symlink
dvc config cache.shared group                               # (将来 team 対応時)
```

**注意点**:
- `cache.type symlink` は workspace 側のファイルを symlink にするため **DVC が自動で read-only 化** (`dvc unprotect` で解除)。編集は `dvc unprotect <path>` 後に行う。
- `dvc gc` は cache を跨ぐ参照を壊すリスク。`--projects` で対象限定。
- cache のパスは `.dvc/config` に書かれる（git 追跡）。絶対パスは環境依存なので、**`.dvc/config.local` に書き、`.dvc/config` には相対パスかコメントで運用指示のみ**を残す方がポータブル。

### 1d. Push / Pull / 競合

- [dvc push](https://dvc.org/doc/command-reference/push) / [dvc pull](https://doc.dvc.org/command-reference/pull)

**挙動**:
- `dvc push` は「ワークスペースの dvc.lock / *.dvc に現れるが remote に未存在」のコンテンツのみ upload。
- `dvc pull` は逆。
- **content-addressable (md5)** なので同一内容なら上書きで破壊されない。ただし **メタ側（Git 追跡の `dvc.lock`）の衝突** は通常の git merge と同じく手で解決。
- [Issue #8354](https://github.com/iterative/dvc/issues/8354): 複数人が同じ outs を並行更新すると、`dvc.lock` の git merge conflict になる。ローカル開発では push 前に `git pull && dvc pull` を習慣化するのが回避策。

## 2. Similar OSS Project Analysis

### Project A — [iterative/example-get-started](https://github.com/iterative/example-get-started)
- **Relevance**: DVC 公式サンプル。preprocess → featurize → train → evaluate の 4 stage pipeline。
- **Approach**:
  - `src/` にスクリプト、`data/` / `models/` / `metrics.json` を outs 宣言。
  - `params.yaml` をトップに配置、stage 側で `params: [prepare.split, train.seed]` とドット記法で参照。
  - `dvc.yaml` は約 40 行でシンプル。
- **Reusable patterns**:
  - stage ごとに `cmd: python src/xxx.py` の粒度（本プロジェクトの `typer` CLI と同等）。
  - metrics は JSON で `dvc metrics diff` の対象。
- **Pitfalls found**: README 上、外部データ取得を `wget` で `dvc run` するのではなく `dvc import-url` を推奨（本プロジェクトは Kaggle scraper が独自なので参考程度）。

### Project B — [matsui-lab blog: Shared dataset versioning with DVC+S3](https://mti-lab.github.io/blog/2021/03/03/dvc.html)
- **Relevance**: 研究室規模で S3 を DVC remote にし、複数メンバーが push/pull する運用事例（日本語）。
- **Approach**:
  - IAM 作成 → S3 bucket 作成 → `dvc remote add -d s3 s3://bucket/path`。
  - メンバーごとに AWS credentials を `~/.aws/credentials` に置き、config.local を使わない。
- **Reusable patterns**:
  - S3 bucket は **単一 prefix で全プロジェクト共有**、`dvc remote add` の URL に prefix を付けて分離（例: `s3://my-bucket/orbit-wars/`）。
- **Pitfalls found**: `dvc gc` を誤って実行して他メンバーの cache を消した事例。remote 側は `dvc gc -c` が必要な点に注意。

### Project C — [anno-ai: Managing Large ML Datasets with DVC and S3](https://anno-ai.medium.com/mlops-and-data-managing-large-ml-datasets-with-dvc-and-s3-part-1-d5b8f2fb8280)
- **Relevance**: TB スケールのデータに対する DVC 運用。本プロジェクトは 1 GB 弱だが「lake / mart 分離」の視点を借用可能。
- **Reusable patterns**:
  - `dvc import` で別 repo のデータを version ピン止めして引き込む。
  - 大きい outs はディレクトリ単位で DVC 管理し、内部ファイルは DVC 追跡しない（outs 単位の md5 で済む）。
- **Pitfalls found**: S3 lifecycle policy で古い version を自動削除すると DVC の古い commit が pull 不能になる。→ bucket versioning と lifecycle は慎重に。

### Pattern Comparison

| Aspect | 本プロジェクト | example-get-started | matsui-lab | anno-ai |
|--------|----------------|---------------------|------------|---------|
| データ規模 | ~850 MB | < 100 MB | ~10 GB | TB |
| stage 粒度 | category/case ごとに分離 | 4 stage 線形 | 2-3 stage | foreach で多数 |
| remote 認証 | AWS profile (local) | env var | 直書き (非推奨) | CI から OIDC |
| 複数 worktree | ある (本件固有) | なし | なし | CI orchestrator |
| 推奨採用 | stage はカテゴリ別、認証は profile、cache はメインリポ共有 |

## 3. Library/Service Selection

### DVC バージョン
- **推奨**: `dvc[s3]>=3.55` (2026-04 時点の最新安定系)。`s3` extra が `boto3`, `aiobotocore` を引く。
- 依存肥大リスク: `uv.lock` に約 40 パッケージ追加。本プロジェクトは既に `kaggle` / `torch` が入っているので相対的な増加は許容範囲。

### 代替候補

| Candidate | Pros | Cons | 判定 |
|-----------|------|------|-----|
| **DVC + S3** | ✅ Git 親和性、pipeline 再現、無料 / ⚠️ S3 課金、cache 管理 | 要学習 | ⭐採用 |
| Git LFS | ✅ 学習コスト低 / ⚠️ pipeline 管理なし、GH LFS は従量課金で高い | 不適 |
| MLflow Tracking + Artifact Store | ✅ 実験比較 UI / ⚠️ サーバ運用、S3 + DB 必要 | 過剰 |
| 自前 scripts + S3 sync | ✅ 最小 / ⚠️ 再現性・ハッシュ検証が手作業 | 却下 |

💡 推薦理由: DVC は「git と同居するパイプライン + 遠隔 storage」要件に最短で対応し、既存 YAML/typer 構造をほぼそのまま stage 化できる。

## 4. AWS / Terraform Research

### 4a. S3 Bucket 設計

- [k9securityio/terraform-aws-s3-bucket](https://github.com/k9securityio/terraform-aws-s3-bucket) を参考にした least-privilege 設計。
- [Terraform S3 Backend](https://developer.hashicorp.com/terraform/language/backend/s3)

**推奨 bucket 構成**:
```
orbit-wars-dvc-<account_id>/
  remote/           # DVC content-addressable storage (md5/ab/cdef... 形式)
  backup/           # 手動 backup / snapshot (optional)
```

**重要設定**:
- `versioning = Enabled`（DVC push で md5 衝突はしないが、誤削除保護）
- `server_side_encryption_configuration`: AES256 または KMS
- `block_public_access`: 全部 true
- `lifecycle_rule`: **古い version の自動削除は無効** にする（anno-ai の教訓）

### 4b. IAM Policy (least privilege)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DvcRemoteListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::orbit-wars-dvc-<account_id>"
    },
    {
      "Sid": "DvcRemoteObject",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::orbit-wars-dvc-<account_id>/remote/*"
    }
  ]
}
```

- `s3:DeleteObject` は `dvc gc -c` 時にのみ必要。開発者ロールに付けて管理者のみ `dvc gc -c` 実行する運用で十分。

### 4c. Terraform state backend のブートストラップ

- DVC remote bucket 自体を Terraform で管理するなら、**state bucket** も別途必要（chicken-and-egg）。
- 本プロジェクトは `infra/environment/dev/` 配下のみ存在し Terraform 稼働実績なし。
- **推奨**: 初回は local state で bootstrap → state backend の S3 bucket を作成 → `backend "s3"` に切替。

## 5. Research Summary

- **アーキテクチャ推奨**:
  1. `dvc.yaml` を **category 単位でトップレベル 1 本** に集約（`stages` 下に `scrape_kaggle`, `preprocess_imitation_case1`, `train_imitation_case1`, `eval_imitation_case1` ... を展開）。
  2. `params.yaml` はルートに置き、`backend/pipeline/.../configs/*.yaml` は **段階的に params 化**（破壊せず併存）。
  3. Cache はメインリポ側 `/Users/user/project/orbit-wars/.dvc/cache` に固定、`.dvc/config.local` で絶対パスを指定（worktree 共通）。
  4. S3 remote: `s3://orbit-wars-dvc-<account>/remote`、認証は AWS profile `orbit-wars`。
  5. Terraform module: `infra/module/application/dvc_remote/` に bucket + IAM policy + ユーザーロール。
- **Kaggle 提出物 `weights.pt` の完全 DVC 化**: ユーザー確定済み。
  - 提出前に `dvc pull` でローカルに実体化 → packager が tar.gz 化 → Kaggle へ upload。
  - CI/CD submit にも `dvc pull` ステップを組み込む必要（今回は scope 外、将来タスク）。
- **採用しないもの**: MLflow, Git LFS, cloud-versioned remote (DVC のクラウドバージョニング機能、worktree切替で merge conflict を起こすため)。

## Sources

- [DVC S3 Remote Storage](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3)
- [DVC Get Started: Data Pipelines](https://doc.dvc.org/start/data-pipelines/data-pipelines)
- [dvc.yaml Files](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files)
- [DVC: Share a Cache](https://doc.dvc.org/user-guide/how-to/share-a-dvc-cache)
- [dvc push](https://dvc.org/doc/command-reference/push), [dvc pull](https://doc.dvc.org/command-reference/pull), [dvc remote modify](https://dvc.org/doc/command-reference/remote/modify)
- [iterative/dvc Issue #8354 — merge conflicts with cloud versioned remotes](https://github.com/iterative/dvc/issues/8354)
- [Matsui-lab: Versioning shared dataset with DVC and S3](https://mti-lab.github.io/blog/2021/03/03/dvc.html)
- [Anno.ai: Managing Large ML Datasets with DVC and S3](https://anno-ai.medium.com/mlops-and-data-managing-large-ml-datasets-with-dvc-and-s3-part-1-d5b8f2fb8280)
- [k9securityio/terraform-aws-s3-bucket (least-privilege reference)](https://github.com/k9securityio/terraform-aws-s3-bucket)
- [Terraform S3 Backend](https://developer.hashicorp.com/terraform/language/backend/s3)
