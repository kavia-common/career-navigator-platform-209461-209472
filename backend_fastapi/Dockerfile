# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (optional: for sqlite3 & build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential sqlite3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src /app/src

# Expose port
EXPOSE 8000

# Ensure data directory exists for SQLite volume mount
RUN mkdir -p /data

# Default to uvicorn
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}"]
