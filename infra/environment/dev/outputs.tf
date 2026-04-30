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
  description = "ECR repository URL for the Vast.ai runtime base image."
  value       = module.ecr_runtime.repository_url
}

output "ecr_runtime_push_user_name" {
  description = "IAM user authorized to push to the runtime ECR repository."
  value       = module.ecr_runtime.push_user_name
}
