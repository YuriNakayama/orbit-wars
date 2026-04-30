variable "prefix" {
  description = "Resource name prefix"
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

variable "public_ecr_repository_uri" {
  description = "Public ECR Gallery URI (Vast.ai が pull する先)"
  type        = string
}

variable "public_ecr_repository_arn" {
  description = "Public ECR repository ARN"
  type        = string
}

variable "dvc_bucket_name" {
  description = "DVC remote S3 bucket name (read-only access for cache hydration)"
  type        = string
}
