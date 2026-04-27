# infra/environment/dev

DVC remote (S3) を払い出す dev 環境の Terraform root module。

## 初回セットアップ

1. AWS CLI で使うプロファイル (例: `orbit-wars-admin`) に管理権限を付与した IAM を用意し、`~/.aws/credentials` に配置する。
2. 本ディレクトリに `terraform.tfvars` を作成 (`terraform.tfvars.example` をコピーし bucket 名を差し替え)。
   - `dvc_bucket_name` はグローバル一意。推奨フォーマット: `orbit-wars-dvc-<AWS_ACCOUNT_ID>`
3. `AWS_PROFILE=orbit-wars-admin terraform init`
4. `AWS_PROFILE=orbit-wars-admin terraform plan`
5. ユーザー承認後に `AWS_PROFILE=orbit-wars-admin terraform apply`

## 出力値から DVC を設定する

apply 完了後:

```bash
# 出力された access key を profile に登録
AWS_PROFILE=orbit-wars aws configure set aws_access_key_id "$(terraform output -raw dvc_iam_access_key_id)"
AWS_PROFILE=orbit-wars aws configure set aws_secret_access_key "$(terraform output -raw dvc_iam_secret_access_key)"

# .dvc/config の bucket URL を更新 (リポジトリルートから)
dvc remote modify s3 url "s3://$(terraform output -raw dvc_bucket_name)/remote"
```

## State 管理

初回は **local state** で bootstrap する (`versions.tf` に backend ブロック無し)。
apply 後、state 専用 bucket (`orbit-wars-tfstate-<account>`) を別途 Terraform で作成してから
`backend "s3"` ブロックを追加し `terraform init -migrate-state` で移行する。

## 削除

```
AWS_PROFILE=orbit-wars-admin terraform destroy
```

**注意**: bucket 内オブジェクトが残っていると destroy 失敗する。事前に `aws s3 rm s3://<bucket> --recursive` が必要。
