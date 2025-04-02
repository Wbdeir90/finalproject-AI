import os
from google.cloud import aiplatform
import logging

# Set your project details
PROJECT_ID = "finalproject-1234567"
REGION = "us-central1"
GCS_BUCKET = "gs://groupfinal-central-staging/output"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Google Cloud credentials
GCP_CREDENTIALS_PATH = "C:\\Users\\wafaa\\gcp-creds\\finalproject-1234567-e5617b2836cb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH

# Fetch the custom training image
TRAINING_IMAGE = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"

if not TRAINING_IMAGE:
    raise ValueError("The environment variable 'TRAINING_IMAGE' is not set. Please set it to the correct Docker image URI.")

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=GCS_BUCKET
)

def create_custom_training_job():
    """Creates and runs a custom training job on Vertex AI."""
    
    job = aiplatform.CustomTrainingJob(
        display_name="spam-classifier-job",
        script_path="train.py",
        container_uri=TRAINING_IMAGE,
        requirements=["scikit-learn", "pandas", "tensorflow"],
    )

    model = job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        args=["--batch_size", "64", "--epochs", "10"],
        base_output_dir=f"{GCS_BUCKET}/output",
    )

    print("Training job submitted successfully!")
    return model

if __name__ == "__main__":
    create_custom_training_job()
