# Create CloudWatch Log Group
resource "aws_cloudwatch_log_group" "toxic-comments" {
  name              = "/aws/ec2/toxic-comments-app"
  retention_in_days = 7

  tags = {
    Environment = "production"
    Project     = "Toxic-Comments-AWS"
    ManagedBy   = "Terraform"
  }
}


