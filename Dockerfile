# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Install Apache Beam (needed for GCP)
RUN pip install apache-beam[gcp] pytest gunicorn

# Copy the application code
COPY . .

# Ensure the container runs on port 8080
EXPOSE 8080

# Run unit tests before starting the app
RUN pytest tests/

# Set the entrypoint for Cloud Run
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Test:app"]
