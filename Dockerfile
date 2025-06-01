# Use a standard Python base image (for example, python:3.9-slim) as the base.
FROM python:3.9-slim as base

# Test stage
FROM base as test
# Copy the project files
COPY . .
# Install main and dev requirements (from requirements.txt and requirements-dev.txt)
RUN pip install -r requirements.txt -r requirements-dev.txt
# Install the package in development mode
RUN pip install -e .
# Install additional test dependencies
RUN pip install pytest-cov pytest-asyncio

# Production stage
FROM base
# Copy the project files
COPY . .
# Install only main requirements (from requirements.txt)
RUN pip install -r requirements.txt
# Install the package in development mode
RUN pip install -e .
# Set the entrypoint
ENTRYPOINT ["python"] 