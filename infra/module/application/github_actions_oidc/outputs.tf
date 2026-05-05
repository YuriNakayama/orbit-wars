output "role_arn" {
  description = "IAM role ARN assumed by GitHub Actions via OIDC."
  value       = aws_iam_role.github_actions.arn
}

output "role_name" {
  description = "IAM role name."
  value       = aws_iam_role.github_actions.name
}
