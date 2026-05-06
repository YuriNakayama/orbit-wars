variable "prefix" {
  description = "Resource name prefix applied to the IAM role (e.g. orbit-wars-dev)."
  type        = string
}

variable "github_repo" {
  description = "GitHub repository allowed to assume this role, in `owner/repo` form."
  type        = string
  default     = "YuriNakayama/orbit-wars"
}

variable "ecr_repository_arns" {
  description = "ARNs of ECR repositories the role can push to."
  type        = list(string)
}

variable "ecs_cluster_arn" {
  description = "ARN of the ECS cluster the role can run tasks on."
  type        = string
}

variable "ecs_task_role_arns" {
  description = "Task / execution role ARNs that the role is allowed to PassRole to ECS."
  type        = list(string)
}

variable "ecs_log_group_arn" {
  description = "CloudWatch log group ARN (with `:*` suffix) the role can read for ECS task logs."
  type        = string
}

variable "dvc_bucket_arn" {
  description = "DVC remote S3 bucket ARN; the role gets read access so docker build can sync the DVC cache into the runtime image."
  type        = string
}
