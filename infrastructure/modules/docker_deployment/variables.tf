variable "instance_ip" {
  description = "Public IP of the EC2 instance"
  type        = string
}

variable "private_key" {
  description = "Private key for SSH connection"
  type        = string
  sensitive   = true
}

variable "file_sources" {
  description = "List of files to copy to the instance"
  type = list(object({
    source      = string
    destination = string
  }))
  default = []
}

variable "docker_setup_commands" {
  description = "Commands to set up Docker"
  type        = list(string)
}

variable "build_and_run_commands" {
  description = "Commands to build and run the Docker container"
  type        = list(string)
}
