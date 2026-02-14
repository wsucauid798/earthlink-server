# EarthLink Server

A symbolic virtual world server, built from real Earth data.

EarthLink simulates a living world where autonomous agents explore, learn, and interact with a geographically accurate representation of real places, real connections, real weather.

## Prerequisites

- Python 3.12+
- PostgreSQL (with the EarthLink database populated)
- Redis (for shared state)
- ChromaDB (for vector memory)

## Setup

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS / Linux

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

## Running

```bash
# Start the server
uvicorn main:app --app-dir src --reload
```

The API is served at `http://localhost:8000` by default. Interactive docs at `/docs`.

## Testing

```bash
pytest
```

## License

All rights reserved.
