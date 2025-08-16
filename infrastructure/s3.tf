# This resource defines a random string to be appended to the S3 bucket name.
# This helps ensure the bucket name is globally unique.
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket" "toxic_comments_assets" {
  bucket        = "${var.s3_bucket}-${random_string.bucket_suffix.result}"
  force_destroy = true
}

# This output block will display the S3 bucket name
output "s3_bucket_name" {
  description = "The name of the S3 bucket for storing assets."
  value       = aws_s3_bucket.toxic_comments_assets.bucket
}