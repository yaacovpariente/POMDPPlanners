# Use a standard Python base image (for example, python:3.10-slim) as the base.
FROM python:3.10-slim AS base

# Test stage
FROM base AS test

WORKDIR /app

RUN apt-get update && apt-get install -y libatomic1 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -e ".[dev,docs]"

# Production stage
FROM base

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -e .

ENTRYPOINT ["python"] 