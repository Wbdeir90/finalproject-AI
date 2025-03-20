# Use TensorFlow as a base image for machine learning
FROM tensorflow/tensorflow:2.15.0

# Set the working directory in the container
WORKDIR /trainer

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training scripts
COPY . .

# Ensure GCP credentials are in place
COPY gcp-creds/finalproject-1234567-e5617b2836cb.json /root/gcp-creds/finalproject-1234567-e5617b2836cb.json

# Set environment variable for GCP credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/gcp-creds/finalproject-1234567-e5617b2836cb.json

# Run the training script
CMD ["python", "train.py"]
