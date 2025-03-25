# Stage 1: Build dependencies
FROM python:3.9 AS build

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# Expose the port Cloud Run expects
EXPOSE 8080

# Copy dependencies and app files
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app

# Set environment variable for Cloud Run
ENV PORT=8080

# Start the app
CMD ["python", "app.py"]
