# CodeBuild project: orbit-wars Vast.ai runtime image を ECR に build & push する.
# トリガ: GitHub の `pyproject.toml` / `uv.lock` / `infra/runtime/**` 変更時、
# あるいは手動 (terraform 経由 / GitHub Actions / `aws codebuild start-build`).
#
# 利点:
#  - ローカル Docker (8GB RAM, builder cache 21GB) に依存しない
#  - GitHub Actions secrets を使わずに OIDC で AWS 認証
#  - Terraform で project 設定 + IAM を完全管理

resource "aws_codebuild_project" "runtime" {
  name         = "${var.prefix}-runtime"
  description  = "Build & push orbit-wars Vast.ai runtime image to ECR"
  service_role = aws_iam_role.codebuild.arn

  source {
    type            = "GITHUB"
    location        = var.github_repo_url
    git_clone_depth = 1
    buildspec       = "infra/runtime/buildspec.yml"
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
}

resource "aws_iam_role_policy" "codebuild" {
  name   = "${var.prefix}-runtime-codebuild-inline"
  role   = aws_iam_role.codebuild.id
  policy = data.aws_iam_policy_document.codebuild_inline.json
}
