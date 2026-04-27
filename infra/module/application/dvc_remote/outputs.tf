output "bucket_name" {
  description = "DVC remote S3 bucket name."
  value       = aws_s3_bucket.dvc_remote.bucket
}

output "bucket_arn" {
  description = "DVC remote S3 bucket ARN."
  value       = aws_s3_bucket.dvc_remote.arn
}

output "iam_user_name" {
  description = "IAM user created for DVC remote access."
  value       = aws_iam_user.dvc_user.name
}

output "iam_access_key_id" {
  description = "Access key ID for the DVC IAM user. Configure this in ~/.aws/credentials."
  value       = aws_iam_access_key.dvc_user.id
  sensitive   = true
}

output "iam_secret_access_key" {
  description = "Secret access key for the DVC IAM user."
  value       = aws_iam_access_key.dvc_user.secret
  sensitive   = true
}
