output "dvc_bucket_name" {
  description = "Configured value for `dvc remote modify s3 url s3://<bucket>/remote`."
  value       = module.dvc_remote.bucket_name
}

output "dvc_iam_user_name" {
  description = "IAM user dedicated to DVC remote access."
  value       = module.dvc_remote.iam_user_name
}

output "dvc_iam_access_key_id" {
  description = "Access key ID. Add to ~/.aws/credentials under profile `orbit-wars`."
  value       = module.dvc_remote.iam_access_key_id
  sensitive   = true
}

output "dvc_iam_secret_access_key" {
  description = "Secret access key (sensitive)."
  value       = module.dvc_remote.iam_secret_access_key
  sensitive   = true
}

output "ecr_runtime_repository_url" {
  description = "Private ECR repository URL (push from CI/dev only)."
  value       = module.ecr_runtime.repository_url
}

output "ecr_runtime_public_uri" {
  description = "Public ECR Gallery URI used by Vast.ai for anonymous docker pull."
  value       = module.ecr_runtime.public_repository_uri
}

output "ecr_runtime_push_user_name" {
  description = "IAM user authorized to push to the runtime ECR repository."
  value       = module.ecr_runtime.push_user_name
}

output "ecr_runtime_push_access_key_id" {
  description = "Access key ID for the ECR push user."
  value       = module.ecr_runtime.push_access_key_id
  sensitive   = true
}

output "ecr_runtime_push_secret_access_key" {
  description = "Secret access key for the ECR push user (sensitive)."
  value       = module.ecr_runtime.push_secret_access_key
  sensitive   = true
}

output "runtime_codebuild_project_name" {
  description = "Trigger via: aws codebuild start-build --project-name <this>"
  value       = module.runtime_codebuild.project_name
}

output "runtime_codebuild_source_bucket" {
  description = "S3 bucket to upload source zip before triggering CodeBuild."
  value       = module.runtime_codebuild.source_bucket_name
}

output "github_actions_role_arn" {
  description = "IAM role ARN for build-push.yml. Set this as repo secret AWS_ROLE_ARN."
  value       = module.github_actions_oidc.role_arn
}
