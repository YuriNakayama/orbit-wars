variable "prefix" {
  description = "Resource name prefix"
  type        = string
}

variable "github_repo_url" {
  description = "GitHub repository URL (HTTPS clone url)"
  type        = string
}

variable "ecr_repository_url" {
  description = "ECR repository URL (passed to docker push)"
  type        = string
}

variable "ecr_repository_arn" {
  description = "ECR repository ARN (used in IAM resource scoping)"
  type        = string
}

variable "dvc_bucket_name" {
  description = "DVC remote S3 bucket name (read-only access for cache hydration)"
  type        = string
}
