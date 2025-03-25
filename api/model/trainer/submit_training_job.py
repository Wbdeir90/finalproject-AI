from google.cloud import aiplatform

# Initialize AI Platform job
job = aiplatform.CustomPythonPackageTrainingJob(
    display_name="spam-classifier-training",
    python_package_gcs_uri="gs://groupfinal-central/spam_classifier/",  # Update with your actual GCS path
    python_module_name="trainer.train",  # This should match your training script
    container_uri="us-central1-docker.pkg.dev/finalproject-1234567/my-repo/spam-classifier-api:latest"
)

# Run the training job
job.run(
    replica_count=1,  # Adjust if needed
    machine_type="n1-standard-4",  # Choose appropriate compute resources
    args=[]  # Pass additional training arguments if necessary
)
