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
