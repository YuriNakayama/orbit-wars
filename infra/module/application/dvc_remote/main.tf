resource "aws_s3_bucket" "dvc_remote" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_versioning" "dvc_remote" {
  bucket = aws_s3_bucket.dvc_remote.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_remote" {
  bucket = aws_s3_bucket.dvc_remote.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_remote" {
  bucket                  = aws_s3_bucket.dvc_remote.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Kaggle Kernel interactive session の S3 channel オブジェクトは ephemeral。
# 取り残されたまま課金されないよう、7 日経過で expire させる。
resource "aws_s3_bucket_lifecycle_configuration" "dvc_remote" {
  bucket = aws_s3_bucket.dvc_remote.id

  rule {
    id     = "kaggle_interactive_expiration"
    status = "Enabled"

    filter {
      prefix = "kaggle_interactive/"
    }

    expiration {
      days = 7
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

resource "aws_iam_user" "dvc_user" {
  name = "${var.prefix}-dvc-user"
}

resource "aws_iam_access_key" "dvc_user" {
  user = aws_iam_user.dvc_user.name
}

data "aws_iam_policy_document" "dvc_remote_rw" {
  statement {
    sid       = "ListBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.dvc_remote.arn]
  }

  statement {
    sid    = "ReadWriteObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.dvc_remote.arn}/remote/*"]
  }

  # Kaggle Kernel interactive mode: S3 を双方向 command channel として使うため、
  # inbox を取り出し後に削除して再利用する必要がある。DVC append-only な
  # ``remote/*`` 領域とは独立の prefix で、Delete を含めた完全な CRUD を許可する。
  statement {
    sid    = "KaggleInteractiveChannel"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.dvc_remote.arn}/kaggle_interactive/*"]
  }
}

resource "aws_iam_user_policy" "dvc_remote_rw" {
  name   = "${var.prefix}-dvc-remote-rw"
  user   = aws_iam_user.dvc_user.name
  policy = data.aws_iam_policy_document.dvc_remote_rw.json
}
