FROM python:3.13-slim

LABEL org.opencontainers.image.title="earthlink-server"
LABEL org.opencontainers.image.version="0.0.1"
LABEL org.opencontainers.image.description="A symbolic virtual world — the United Kingdom, built from real Earth data"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy all application code first (source needed for editable-style install)
COPY . .

# Install Python package and all dependencies
RUN pip install --no-cache-dir .

# Create data directory for cached downloads
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
