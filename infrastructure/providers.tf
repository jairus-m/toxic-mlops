# This block configures Terraform to use the AWS provider.
# It specifies the source of the provider (hashicorp/aws) and a version constraint.
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Use a recent version of the AWS provider
    }
  }
}

# This block configures the AWS provider itself.
# You should change the region to match your AWS lab environment if it's not us-east-1.
provider "aws" {
  region = var.aws_region
}
