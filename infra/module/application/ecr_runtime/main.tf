# Vast.ai 学習ノードが pull する Docker base image を ECR で管理する.
# image には orbit-wars/backend の依存ライブラリ (torch + CUDA wheels 等) を
# 焼き込んでおき、Vast インスタンス側の uv sync を skip 〜 短縮する.

resource "aws_ecr_repository" "runtime" {
  name                 = "${var.prefix}-runtime"
  image_tag_mutability = "MUTABLE" # latest tag を上書きするため

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Vast.ai インスタンスが pull するため public read を許可する IAM policy.
# image の中身は依存ライブラリのみで秘密情報は含まないので公開でも安全.
resource "aws_ecr_repository_policy" "public_pull" {
  repository = aws_ecr_repository.runtime.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowPublicPull"
        Effect    = "Allow"
        Principal = "*"
        Action = [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
      },
    ]
  })
}

# 古い image を保持しすぎないため lifecycle policy で 10 個まで残す.
resource "aws_ecr_lifecycle_policy" "keep_recent" {
  repository = aws_ecr_repository.runtime.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      },
    ]
  })
}

# CI / 開発者ローカルから push するための IAM user. credentials は
# `aws iam create-access-key --user-name <name>` で発行し、
# GitHub Actions secrets / 開発者の ~/.aws/credentials に置く運用.
resource "aws_iam_user" "ecr_push" {
  name = "${var.prefix}-ecr-push"
}

resource "aws_iam_user_policy" "ecr_push" {
  name = "${var.prefix}-ecr-push-policy"
  user = aws_iam_user.ecr_push.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GetAuthToken"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
        ]
        Resource = "*"
      },
      {
        Sid    = "PushPullThisRepo"
        Effect = "Allow"
        Action = [
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
        Resource = aws_ecr_repository.runtime.arn
      },
    ]
  })
}
