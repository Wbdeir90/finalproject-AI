provider "google" {
  project = var.project_id
  region  = "us-central1"
}

variable "project_id" {
  description = "The Google Cloud Project ID"
  type        = string
}

variable "image_tag" {
  description = "The image tag for the Cloud Run service"
  type        = string
}

resource "google_artifact_registry_repository" "repo" {
  project       = var.project_id
  location      = "us-central1"
  repository_id = "my-repo"
  format        = "DOCKER"
}

resource "google_cloud_run_service" "spam_api" {
  name     = "spam-classifier-api"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us-central1-docker.pkg.dev/${var.project_id}/my-repo/spam-classifier-api:${var.image_tag}"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}
