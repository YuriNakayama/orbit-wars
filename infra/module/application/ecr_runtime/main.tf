# Vast.ai 学習ノードが pull する Docker base image を ECR で管理する.
# image には orbit-wars/bot の依存ライブラリ (torch + CUDA wheels 等) を
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

# Public ECR Gallery (us-east-1) で同名 repository を払い出す.
# Vast.ai インスタンスが匿名で `docker pull` できるよう public 化.
resource "aws_ecrpublic_repository" "runtime" {
  provider        = aws.us_east_1
  repository_name = "${var.prefix}-runtime"

  catalog_data {
    about_text        = "Vast.ai runtime base image for orbit-wars (uv + DVC cache baked)."
    architectures     = ["x86-64"]
    operating_systems = ["Linux"]
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
          "ecr-public:GetAuthorizationToken",
          "sts:GetServiceBearerToken",
        ]
        Resource = "*"
      },
      {
        Sid    = "PushPullPrivateRepo"
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
      {
        Sid    = "PushPublicRepo"
        Effect = "Allow"
        Action = [
          "ecr-public:BatchCheckLayerAvailability",
          "ecr-public:CompleteLayerUpload",
          "ecr-public:InitiateLayerUpload",
          "ecr-public:PutImage",
          "ecr-public:UploadLayerPart",
          "ecr-public:DescribeRepositories",
          "ecr-public:DescribeImages",
        ]
        Resource = aws_ecrpublic_repository.runtime.arn
      },
    ]
  })
}

# Push 用 access key を Terraform で発行する. 発行後は出力値を ~/.aws/credentials
# (profile=orbit-wars-ecr-push) に登録する.
resource "aws_iam_access_key" "ecr_push" {
  user = aws_iam_user.ecr_push.name
}
