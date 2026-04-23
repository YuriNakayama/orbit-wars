terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Initial bootstrap uses local state. After the first apply creates a state bucket,
  # switch to `backend "s3"` (see infra/environment/dev/README.md).
}
