output "repository_url" {
  description = "Private ECR repository URL (push only, requires AWS auth)"
  value       = aws_ecr_repository.runtime.repository_url
}

output "public_repository_uri" {
  description = "Public ECR Gallery URI for anonymous docker pull (Vast.ai 等から)"
  value       = aws_ecrpublic_repository.runtime.repository_uri
}

output "public_repository_arn" {
  description = "Public ECR repository ARN (for IAM resource scoping)"
  value       = aws_ecrpublic_repository.runtime.arn
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = aws_ecr_repository.runtime.arn
}

output "repository_name" {
  description = "ECR repository name"
  value       = aws_ecr_repository.runtime.name
}

output "push_user_name" {
  description = "IAM user name authorized to push images"
  value       = aws_iam_user.ecr_push.name
}

output "push_access_key_id" {
  description = "Access key ID for the push user. Register under AWS profile."
  value       = aws_iam_access_key.ecr_push.id
}

output "push_secret_access_key" {
  description = "Secret access key for the push user (sensitive)."
  value       = aws_iam_access_key.ecr_push.secret
  sensitive   = true
}
