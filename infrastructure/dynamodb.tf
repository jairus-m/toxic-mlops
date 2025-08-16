# DynamoDB table for prediction logs
resource "aws_dynamodb_table" "prediction_logs" {
  name         = "prediction_logs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "partition_key"
  range_key    = "sort_key"

  attribute {
    name = "partition_key"
    type = "S"
  }

  attribute {
    name = "sort_key"
    type = "S"
  }

  # Global Secondary Index for querying by timestamp
  global_secondary_index {
    name            = "timestamp-index"
    hash_key        = "log_date"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  attribute {
    name = "log_date"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  tags = {
    Name        = "Prediction Logs Table"
    Project     = "Toxic-Comments-AWS"
    ManagedBy   = "Terraform"
    Environment = "production"
  }
}

# Output the table name for use in other resources
output "dynamodb_table_name" {
  description = "Name of the DynamoDB table for prediction logs"
  value       = aws_dynamodb_table.prediction_logs.name
}

output "dynamodb_table_arn" {
  description = "ARN of the DynamoDB table for prediction logs"
  value       = aws_dynamodb_table.prediction_logs.arn
}