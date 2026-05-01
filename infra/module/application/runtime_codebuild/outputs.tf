output "project_name" {
  description = "CodeBuild project name (use with aws codebuild start-build)"
  value       = aws_codebuild_project.runtime.name
}

output "project_arn" {
  description = "CodeBuild project ARN"
  value       = aws_codebuild_project.runtime.arn
}

output "source_bucket_name" {
  description = "S3 bucket where the source zip is uploaded before each build."
  value       = aws_s3_bucket.source.id
}

output "source_object_key" {
  description = "S3 key for the source zip (CodeBuild source location)"
  value       = "source/orbit-wars-source.zip"
}
