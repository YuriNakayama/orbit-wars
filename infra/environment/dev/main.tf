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
