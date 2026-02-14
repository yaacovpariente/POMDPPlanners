# Use a standard Python base image (for example, python:3.10-slim) as the base.
FROM python:3.10-slim AS base

# Test stage
FROM base AS test

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt -r requirements-dev.txt
RUN pip install -e .[docs]

# Production stage
FROM base

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

ENTRYPOINT ["python"] 