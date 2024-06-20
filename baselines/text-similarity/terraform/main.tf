# bva-citation.json is a Google Credentials Keyfile for authentication
provider "google" {
  credentials = file("~/bva-citation.json")
  project     = "bva-citation"
  region      = "us-east4"
  zone        = "us-east4-c"
}

provider "google-beta" {
  credentials = file("~/bva-citation.json")
  project     = "bva-citation"
  region      = "us-east4"
  zone        = "us-east4-c"
}


resource "google_compute_instance" "vm_instance" {
  name         = "dev-cpu"
  machine_type = "e2-standard-8"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-1804-lts"
    }
  }

  attached_disk {
    source = "data-disk2"
    device_name = "data-disk"
  }

  network_interface {
    # A default network is created for all GCP projects
    network = "default"
    access_config {
    }
  }

  ## Preemptible instance to save cost
  #scheduling {
  #  preemptible = true
  #  automatic_restart = false
  #}

  # Startup Script
  metadata_startup_script = "${data.template_file.install.rendered}"
}

data "template_file" "install" {
  template = "${file("./install.sh")}"
}

# Open port for Tensorboard
resource "google_compute_firewall" "default" {
  name    = "tensorboard-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["6006"]
  }
}
