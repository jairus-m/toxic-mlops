resource "null_resource" "deploy" {
  triggers = {
    always_run = "${timestamp()}"
  }

  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = var.private_key
    host        = var.instance_ip
  }

  # Wait for SSH to be ready
  provisioner "remote-exec" {
    inline = ["echo 'SSH is ready'"]
  }

  # Create the destination directory structure
  provisioner "remote-exec" {
    inline = ["mkdir -p /home/ec2-user/app/src"]
  }
}

resource "null_resource" "copy_files" {
  for_each = { for f in var.file_sources : f.destination => f }

  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = var.private_key
    host        = var.instance_ip
  }

  provisioner "file" {
    source      = each.value.source
    destination = each.value.destination
  }

  depends_on = [null_resource.deploy]
}

resource "null_resource" "run_docker" {
  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = var.private_key
    host        = var.instance_ip
  }

    # Build and run the Docker container
  provisioner "remote-exec" {
    inline = concat(
      ["set -x"],
      var.docker_setup_commands,
      ["sudo systemctl status docker"],
      ["cd /home/ec2-user/app", "ls -la", "sudo docker --version"],
      var.build_and_run_commands,
      ["sudo docker ps -a"]
    )
  }

  depends_on = [null_resource.copy_files]
}
