# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure GCP credentials are in place
COPY gcp-creds/finalproject-1234567-e5617b2836cb.json /root/gcp-creds/finalproject-1234567-e5617b2836cb.json

# Set environment variable for GCP credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/gcp-creds/finalproject-1234567-e5617b2836cb.json

# Install Apache Beam for GCP (if needed)
RUN pip install apache-beam[gcp]

# Expose port 8080
EXPOSE 8080

# Set the working directory for running the app
WORKDIR /app/api

# Run the application
CMD ["python", "app.py"]