# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables for Ray
ENV RAY_DEDICATED=0
ENV RAY_DISABLE_DASHBOARD=1
ENV RAY_DISABLE_METRICS_EXPORT=1
ENV RAY_redis_password=5241590000000000

# Create a temporary directory for Ray
RUN mkdir -p /tmp/ray

# Set the entrypoint
ENTRYPOINT ["python"] 