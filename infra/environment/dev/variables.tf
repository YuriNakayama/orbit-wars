variable "aws_region" {
  description = "AWS region for dev resources."
  type        = string
  default     = "ap-northeast-1"
}

variable "dvc_bucket_name" {
  description = "Globally-unique S3 bucket name for the DVC remote (e.g. orbit-wars-dvc-<account_id>)."
  type        = string
}

variable "resource_prefix" {
  description = "Prefix applied to IAM resources."
  type        = string
  default     = "orbit-wars-dev"
}
