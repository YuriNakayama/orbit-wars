provider "aws" {
  region = var.aws_region

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
  prefix      = var.resource_prefix
}

# Vast.ai 学習ノードが pull する Docker base image (依存焼込み済) を ECR で管理.
module "ecr_runtime" {
  source = "../../module/application/ecr_runtime"
  prefix = var.resource_prefix
}

# CodeBuild project: GitHub の変更検知 / 手動 trigger で image を build & push.
# ローカル Docker (容量逼迫) に依存しないクラウド側 build pipeline.
module "runtime_codebuild" {
  source             = "../../module/application/runtime_codebuild"
  prefix             = var.resource_prefix
  ecr_repository_url = module.ecr_runtime.repository_url
  ecr_repository_arn = module.ecr_runtime.repository_arn
  dvc_bucket_name    = var.dvc_bucket_name
}
