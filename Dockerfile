FROM python:3.13-slim

LABEL org.opencontainers.image.title="earthlink-server"
LABEL org.opencontainers.image.version="0.0.1"
LABEL org.opencontainers.image.description="A symbolic virtual world — the United Kingdom, built from real Earth data"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first — this layer is cached until pyproject.toml changes.
# Code changes don't trigger a full dependency reinstall (torch alone is 915MB).
COPY pyproject.toml .

# Create a minimal stub so pip can resolve the package metadata without full source.
RUN mkdir -p src/world src/db src/api src/data_acquisition src/agents src/adapters && \
    touch src/world/__init__.py src/db/__init__.py src/api/__init__.py \
          src/data_acquisition/__init__.py src/agents/__init__.py src/adapters/__init__.py

# Install all dependencies (cached as long as pyproject.toml is unchanged)
RUN pip install --no-cache-dir .

# Now copy the actual application code — only this layer rebuilds on code changes
COPY . .

# Create data directory for cached downloads
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
