FROM python:3.13-slim

LABEL org.opencontainers.image.title="earthlink-server"
LABEL org.opencontainers.image.version="0.0.2"
LABEL org.opencontainers.image.description="A symbolic virtual world — built from real Earth data"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency installation (strong cache) ---
# Extract only the dependency list from pyproject.toml into requirements.txt.
# Metadata changes (version, description) do NOT bust the pip cache —
# only adding/removing/changing an actual dependency triggers a reinstall.
COPY pyproject.toml .
RUN python -c "\
import tomllib, pathlib; \
data = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); \
deps = data.get('project', {}).get('dependencies', []); \
pathlib.Path('requirements.txt').write_text('\n'.join(deps) + '\n')"

# This layer only rebuilds when the actual dependency list changes
RUN pip install -r requirements.txt

# Install the project itself (no deps — they're already installed above)
RUN mkdir -p src/world src/db src/api src/data_acquisition src/agents src/adapters && \
    touch src/world/__init__.py src/db/__init__.py src/api/__init__.py \
          src/data_acquisition/__init__.py src/agents/__init__.py src/adapters/__init__.py
RUN pip install --no-deps .

# Copy only what the app needs at runtime — nothing else enters the image.
# .dockerignore uses a whitelist (starts with *) so this is safe with COPY . .
# but we're explicit here for clarity.
COPY src/ src/
COPY alembic.ini .

# Create data directory for cached downloads
RUN mkdir -p /app/data

# Ensure src/ is on PYTHONPATH for all processes (uvicorn, Ray workers, etc.)
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
