# CodeBuild project: orbit-wars Vast.ai runtime image を ECR に build & push する.
# Source は S3 zip (`source/orbit-wars-source.zip`) を採用. GitHub OAuth 連携が
# 不要になり Terraform 単独で完全管理可能.
# 利点:
#  - ローカル Docker (8GB RAM, builder cache 21GB) に依存しない
#  - GitHub OAuth 連携が不要 (Terraform で完結)
#  - Terraform で project 設定 + IAM を完全管理
# トリガ:
#   1) 開発者ローカル: dev/runtime-build-cloud (zip → s3 cp → start-build)
#   2) 将来 CI 化する場合は GitHub Actions から OIDC で start-build

resource "aws_s3_bucket" "source" {
  bucket = "${var.prefix}-codebuild-source"
}

resource "aws_s3_bucket_versioning" "source" {
  bucket = aws_s3_bucket.source.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "source" {
  bucket = aws_s3_bucket.source.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "source" {
  bucket                  = aws_s3_bucket.source.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "source" {
  bucket = aws_s3_bucket.source.id
  rule {
    id     = "expire-old-zips"
    status = "Enabled"
    filter {}
    expiration {
      days = 14
    }
  }
}

resource "aws_codebuild_project" "runtime" {
  name         = "${var.prefix}-runtime"
  description  = "Build & push orbit-wars Vast.ai runtime image to ECR"
  service_role = aws_iam_role.codebuild.arn

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.source.id}/source/orbit-wars-source.zip"
    buildspec = "infra/runtime/buildspec.yml"
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE" # 8 vCPU / 16 GB / 50 GB scratch
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true # docker build に必要

    environment_variable {
      name  = "ECR_URL"
      value = var.ecr_repository_url
    }
    environment_variable {
      name  = "PUBLIC_ECR_URI"
      value = var.public_ecr_repository_uri
    }
    environment_variable {
      name  = "DVC_S3_BUCKET"
      value = var.dvc_bucket_name
    }
  }

  cache {
    type  = "LOCAL"
    modes = ["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"]
  }

  build_timeout = 60 # minutes
}

# ----- IAM role for CodeBuild -----
data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${var.prefix}-runtime-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

data "aws_iam_policy_document" "codebuild_inline" {
  statement {
    sid    = "Logs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["*"]
  }
  statement {
    sid    = "EcrAuth"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr-public:GetAuthorizationToken",
      "sts:GetServiceBearerToken",
    ]
    resources = ["*"]
  }
  statement {
    sid    = "EcrPush"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
    ]
    resources = [var.ecr_repository_arn]
  }
  statement {
    sid    = "EcrPublicPush"
    effect = "Allow"
    actions = [
      "ecr-public:BatchCheckLayerAvailability",
      "ecr-public:CompleteLayerUpload",
      "ecr-public:InitiateLayerUpload",
      "ecr-public:PutImage",
      "ecr-public:UploadLayerPart",
      "ecr-public:DescribeRepositories",
      "ecr-public:DescribeImages",
    ]
    resources = [var.public_ecr_repository_arn]
  }
  statement {
    sid    = "S3DvcRead"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
    ]
    resources = [
      "arn:aws:s3:::${var.dvc_bucket_name}",
      "arn:aws:s3:::${var.dvc_bucket_name}/*",
    ]
  }
  statement {
    sid    = "S3SourceRead"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.source.arn,
      "${aws_s3_bucket.source.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "codebuild" {
  name   = "${var.prefix}-runtime-codebuild-inline"
  role   = aws_iam_role.codebuild.id
  policy = data.aws_iam_policy_document.codebuild_inline.json
}
