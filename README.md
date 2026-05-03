# EarthLink Server

The Symbolic Virtual World engine. Hosts world state, agent execution, Earth-proxy evidence access, and the API surface.

The server manages a world defined as W = (S, A, T, R, O) -- symbolic state, action set, event-driven transitions, reward topology, and observation interface. Currently instantiated with Earth-derived structure: real places, real weather, real celestial mechanics, and civilisation data proxied live from 21+ open-source adapters.

## Stack

- **Python** with FastAPI + Uvicorn
- **PostgreSQL** -- persistent world state (geography, weather, astronomy, agent positions)
- **Redis** -- ephemeral Earth-proxy cache (TTL-based, LRU eviction)
- **ChromaDB** -- agent semantic memory (vector similarity retrieval)
- **Ray** -- distributed agent actors (parallel tick execution)
- **Docker Compose** -- containerised deployment

## Prerequisites

- Docker and Docker Compose

Or for local development:
- Python 3.12+
- PostgreSQL, Redis, ChromaDB running separately

## Running (Docker)

```bash
docker compose up --build -d
```

Services: server (port 8000), PostgreSQL (5432), Redis (6379), ChromaDB (8001).
WebTransport QUIC world stream is exposed on UDP `4433` by default.

## Running (Local)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
cp .env.example .env          # Edit with your credentials

uvicorn main:app --app-dir src --reload
```

API at `http://localhost:8000`. Interactive docs at `/docs`.
Configure `EARTHLINK_WT_CERT_PATH` and
`EARTHLINK_WT_KEY_PATH` to valid TLS files before startup.

## Testing

```bash
pytest
```

## Research Tools

Evaluation suite for research data collection lives in `tools/research/`. See `tools/research/README.md`.

## License

[MIT](LICENSE)
