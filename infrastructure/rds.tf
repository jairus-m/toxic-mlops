# RDS PostgreSQL instance for MLflow backend storage
resource "aws_db_instance" "mlflow_db" {
  identifier = "mlflow-db"
  
  # Engine configuration
  engine         = "postgres"
  engine_version = "13.15"
  instance_class = "db.t3.micro"
  
  # Storage configuration
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  # Database configuration
  db_name  = "mlflow"
  username = "mlflow"
  password = var.mlflow_db_password
  
  # Network configuration
  vpc_security_group_ids = [aws_security_group.mlflow_db.id]
  publicly_accessible    = false
  
  # Backup and maintenance
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Deletion protection
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name      = "MLflow Database"
    Project   = "Toxic-Comments-AWS"
    ManagedBy = "Terraform"
  }
}

# Output the database endpoint for MLflow server configuration
output "mlflow_db_endpoint" {
  description = "RDS instance endpoint for MLflow database"
  value       = aws_db_instance.mlflow_db.endpoint
  sensitive   = false
}

output "mlflow_db_port" {
  description = "RDS instance port for MLflow database"
  value       = aws_db_instance.mlflow_db.port
  sensitive   = false
}