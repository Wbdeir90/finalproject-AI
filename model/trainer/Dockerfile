# Use official TensorFlow base image
FROM tensorflow/tensorflow:2.15.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /trainer

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Configure GCP credentials
RUN mkdir -p /root/gcp-creds && \
    chmod 700 /root/gcp-creds

COPY gcp-creds/finalproject-1234567-e5617b2836cb.json /root/gcp-creds/
RUN chmod 600 /root/gcp-creds/final