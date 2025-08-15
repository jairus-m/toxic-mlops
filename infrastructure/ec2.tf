# This resource creates a new key pair. Terraform will generate a new key
# and you must save the private key to a file to be able to SSH into the instances.
resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = tls_private_key.rsa.public_key_openssh
}

# This is a helper resource to generate the RSA private key.
resource "tls_private_key" "rsa" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# This local sensitive file resource saves the generated private key to your local machine.
# IMPORTANT: This file should be added to your .gitignore and treated like a password.
resource "local_sensitive_file" "private_key" {
  content  = tls_private_key.rsa.private_key_pem
  filename = "${path.module}/deployer-key.pem"
}

locals {
  docker_setup_commands = [
    "sudo yum install -y docker",
    "sudo systemctl start docker",
    "sudo systemctl enable docker",
    "while [ ! -S /var/run/docker.sock ]; do echo 'Waiting for docker socket...'; sleep 2; done"
  ]

  common_env_vars = [
    "-e APP_ENV=production",
    "-e AWS_ACCESS_KEY_ID=${var.aws_access_key_id}",
    "-e AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}",
    "-e AWS_SESSION_TOKEN=${var.aws_session_token}",
    "-e S3_BUCKET_NAME=${aws_s3_bucket.toxic_comments_assets.bucket}"
  ]

  common_files = [
    {
      source      = "${path.module}/../assets"
      destination = "/home/ec2-user/app/assets"
    },
    {
      source      = "${path.module}/../config.yaml"
      destination = "/home/ec2-user/app/config.yaml"
    },
    {
      source      = "${path.module}/../pyproject.toml"
      destination = "/home/ec2-user/app/pyproject.toml"
    },
    {
      source      = "${path.module}/../src/core"
      destination = "/home/ec2-user/app/src/core"
    }
  ]

  awslogs_config = [
    "--log-driver=awslogs",
    "--log-opt awslogs-region=${var.aws_region}",
    "--log-opt awslogs-group=${aws_cloudwatch_log_group.toxic-comments.name}",
    "--log-opt awslogs-create-group=true",
  ]
}

# Sklearn Model Training EC2 instance
resource "aws_instance" "ml_training" {
  instance_type = "t2.medium"  # Upgraded from t2.micro for ML training memory requirements
  ami           = data.aws_ami.amazon_linux.id

  vpc_security_group_ids = [aws_security_group.backend.id]
  key_name               = aws_key_pair.deployer.key_name
  iam_instance_profile   = "LabInstanceProfile"

  tags = {
    Name      = "Sklearn Model Training Instance"
    Project   = "Toxic-Comments-AWS"
    ManagedBy = "Terraform"
  }
}

# Backend EC2 instance
resource "aws_instance" "backend" {
  depends_on = [
    aws_instance.ml_training,
  ]
  instance_type = "t2.micro"
  ami           = data.aws_ami.amazon_linux.id

  vpc_security_group_ids = [aws_security_group.backend.id]
  key_name               = aws_key_pair.deployer.key_name
  iam_instance_profile   = "LabInstanceProfile"

  tags = {
    Name      = "FastAPI Backend Instance"
    Project   = "Toxic-Comments-AWS"
    ManagedBy = "Terraform"
  }
}

# Frontend EC2 instance
resource "aws_instance" "frontend" {
  depends_on = [
    aws_instance.backend
  ]
  instance_type = "t2.micro"
  ami           = data.aws_ami.amazon_linux.id

  vpc_security_group_ids = [aws_security_group.frontend.id]
  key_name               = aws_key_pair.deployer.key_name
  iam_instance_profile   = "LabInstanceProfile"

  tags = {
    Name      = "Streamlit Frontend Instance"
    Project   = "Toxic-Comments-AWS"
    ManagedBy = "Terraform"
  }
}

# Monitoring EC2 instance
resource "aws_instance" "monitoring" {
  depends_on = [
    aws_instance.frontend
  ]
  instance_type = "t2.micro"
  ami           = data.aws_ami.amazon_linux.id

  vpc_security_group_ids = [aws_security_group.monitoring.id]
  key_name               = aws_key_pair.deployer.key_name
  iam_instance_profile   = "LabInstanceProfile"

  tags = {
    Name      = "Streamlit Monitoring Instance"
    Project   = "Toxic-Comments-AWS"
    ManagedBy = "Terraform"
  }
}

module "ml_training_deployment" {
  source      = "./modules/docker_deployment"
  instance_ip = aws_instance.ml_training.public_ip
  private_key = tls_private_key.rsa.private_key_pem

  file_sources = concat(local.common_files, [
    {
      source      = "${path.module}/../src/sklearn_training"
      destination = "/home/ec2-user/app/src/sklearn_training"
    }
  ])

  docker_setup_commands = local.docker_setup_commands

  build_and_run_commands = [
    "sudo docker build -f src/sklearn_training/Dockerfile -t toxic-comments-training .",
    "sudo docker run -d --name training ${join(" ", local.awslogs_config)} --log-opt awslogs-stream=${aws_instance.ml_training.id}-training ${join(" ", local.common_env_vars)} toxic-comments-training:latest"
  ]
}

module "backend_deployment" {
  source    = "./modules/docker_deployment"
  depends_on = [module.ml_training_deployment]
  instance_ip = aws_instance.backend.public_ip
  private_key = tls_private_key.rsa.private_key_pem

  file_sources = concat(local.common_files, [
    {
      source      = "${path.module}/../src/fastapi_backend"
      destination = "/home/ec2-user/app/src/fastapi_backend"
    }
  ])

  docker_setup_commands = local.docker_setup_commands

  build_and_run_commands = [
    "sudo docker build -f src/fastapi_backend/Dockerfile -t toxic-comments-backend .",
    "sudo docker run -d -p 8000:8000 --restart=always --name backend ${join(" ", local.awslogs_config)} --log-opt awslogs-stream=${aws_instance.backend.id}-backend ${join(" ", local.common_env_vars)} toxic-comments-backend:latest"
  ]
}

module "frontend_deployment" {
  source    = "./modules/docker_deployment"
  depends_on = [module.backend_deployment]
  instance_ip = aws_instance.frontend.public_ip
  private_key = tls_private_key.rsa.private_key_pem

  file_sources = concat(local.common_files, [
    {
      source      = "${path.module}/../src/streamlit_frontend"
      destination = "/home/ec2-user/app/src/streamlit_frontend"
    }
  ])

  docker_setup_commands = local.docker_setup_commands

  build_and_run_commands = [
    "sudo docker build -f src/streamlit_frontend/Dockerfile -t toxic-comments-frontend .",
    "sudo docker run -d -p 8501:8501 --restart=always --name frontend ${join(" ", local.awslogs_config)} --log-opt awslogs-stream=${aws_instance.frontend.id}-frontend ${join(" ", local.common_env_vars)} -e FASTAPI_BACKEND_URL=http://${aws_instance.backend.public_ip}:8000 toxic-comments-frontend:latest"
  ]
}

module "monitoring_deployment" {
  source    = "./modules/docker_deployment"
  depends_on = [module.frontend_deployment]
  instance_ip = aws_instance.monitoring.public_ip
  private_key = tls_private_key.rsa.private_key_pem

  file_sources = concat(local.common_files, [
    {
      source      = "${path.module}/../src/streamlit_monitoring"
      destination = "/home/ec2-user/app/src/streamlit_monitoring"
    }
  ])

  docker_setup_commands = local.docker_setup_commands

  build_and_run_commands = [
    "sudo docker build -f src/streamlit_monitoring/Dockerfile -t toxic-comments-monitoring .",
    "sudo docker run -d -p 8502:8502 --restart=always --name monitoring ${join(" ", local.awslogs_config)} --log-opt awslogs-stream=${aws_instance.monitoring.id}-monitoring ${join(" ", local.common_env_vars)} toxic-comments-monitoring:latest"
  ]
}

# Data source to find the latest Amazon Linux AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Outputs

output "frontend_url" {
  description = "HTTP URL for accessing the Streamlit frontend."
  value       = "http://${aws_instance.frontend.public_ip}:8501"
}

output "monitoring_url" {
  description = "HTTP URL for accessing the Streamlit monitoring."
  value       = "http://${aws_instance.monitoring.public_ip}:8502"
}

output "ssh_command_backend" {
  description = "Command to SSH into the backend instance."
  value       = "ssh -i ${local_sensitive_file.private_key.filename} ec2-user@${aws_instance.backend.public_ip}"
}
