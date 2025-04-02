# Use official Python image from Docker Hub
FROM python:3.9-slim AS builder

# Set the working directory in the builder stage
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libclang-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy only requirements.txt first to leverage layer caching
COPY requirements.txt . 

# Install dependencies in a temporary directory
RUN pip install --no-cache-dir -r requirements.txt -t /app/deps

# Now use a clean Python base image for the final container
FROM python:3.9-slim 

# Set working directory
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /app/deps /usr/local/lib/python3.9/site-packages

# Copy the application code
COPY . /app/

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the app with gunicorn (recommended for production)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
