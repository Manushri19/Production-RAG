# Worker Dockerfile
# Stateless document processing worker (no GPU required)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[worker]"

# Copy application code
COPY meridian/ meridian/

# Run the worker
CMD ["celery", "-A", "meridian.workers.celery_app", "worker", \
     "--loglevel=info", "--concurrency=3"]
