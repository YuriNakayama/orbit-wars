output "repository_url" {
  description = "ECR repository URL (use as docker push/pull target)"
  value       = aws_ecr_repository.runtime.repository_url
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
