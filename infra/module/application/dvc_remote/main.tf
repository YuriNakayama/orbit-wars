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
}

resource "aws_iam_user_policy" "dvc_remote_rw" {
  name   = "${var.prefix}-dvc-remote-rw"
  user   = aws_iam_user.dvc_user.name
  policy = data.aws_iam_policy_document.dvc_remote_rw.json
}
