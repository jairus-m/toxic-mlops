variable "aws_access_key_id" {
  description = "AWS access key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS secret access key"
  type        = string
  sensitive   = true
}

variable "aws_session_token" {
  description = "AWS session token"
  type        = string
  sensitive   = true
}

variable "train_model" {
  description = "If true, runs the full model training pipeline. If false, deploys a pre-existing model."
  type        = bool
  default     = true
}


variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket" {
  description = "Name of the S3 bucket"
  type        = string
  default     = "toxic-comments-s3"
}

variable "mlflow_db_password" {
  description = "Password for the MLflow PostgreSQL database"
  type        = string
  sensitive   = true
  default     = "mlflow123" # Pls dont hack me
}