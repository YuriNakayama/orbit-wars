output "project_name" {
  description = "CodeBuild project name (use with aws codebuild start-build)"
  value       = aws_codebuild_project.runtime.name
}

output "project_arn" {
  description = "CodeBuild project ARN"
  value       = aws_codebuild_project.runtime.arn
}
