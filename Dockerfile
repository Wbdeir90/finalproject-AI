# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Copy the credentials file into the container
COPY gcp-creds/finalproject-1234567-e5617b2836cb.json /root/gcp-creds/finalproject-1234567-e5617b2836cb.json

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application
CMD ["python", "Test.py"]
# Add this line to your Dockerfile
RUN pip install apache-beam[gcp]
