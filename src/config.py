from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (PostgreSQL — world state: geography, weather, positions)
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "earthlink"
    db_password: str = "earthlink"
    db_name: str = "earthlink"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Redis (ephemeral proxy cache — civilisation content with TTL)
    redis_url: str = "redis://localhost:6379/0"

    # Chroma (vector DB — agent semantic memory retrieval)
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    # Pgvector cutover switch (S99). Default OFF: the legacy per-agent
    # ChromaFactStore is retired (S78) and agent memory lives in pgvector
    # (S76/S77), so Chroma must stay out of the per-agent tick path — a wedged
    # Chroma there is what stalled ticks (S66). Defaulting false means the
    # image is safe even where the env var isn't set. Chroma keeps running for
    # its world-semantic-layer role (S79+), just not per-agent. Set true only
    # to temporarily resurrect the legacy per-agent path.
    chroma_enabled: bool = False

    # TEI (Text Embeddings Inference — the single embedding service, S72).
    # Serves BAAI/bge-m3 at 1024 dims (S71). All embedding routes through it;
    # callers degrade gracefully when it is unreachable.
    tei_url: str = "http://localhost:8080"
    tei_embedding_dims: int = 1024  # MUST match db.models.EMBEDDING_DIMS (S71)
    # World semantic layer (S79-S82): Chroma's new role — a shared, TTL'd
    # index over EarthProxy-resolved civilisation content. Independent of the
    # (retired) per-agent chroma path; async + off the tick thread.
    world_semantic_enabled: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # WebTransport (QUIC/HTTP3)
    wt_host: str = "0.0.0.0"
    wt_port: int = 4433
    wt_path: str = "/wt/world"
    wt_cert_path: str = ""
    wt_key_path: str = ""
    # Optional externally reachable WT URL (for reverse proxy / separate hostname).
    # Example: "https://earthlink.yuxilabs.com/wt/world" or "https://wt.earthlink.yuxilabs.com/wt/world"
    wt_public_url: str = ""

    # World — overridable via EARTHLINK_AGENT_COUNT etc.
    # Ray is gated by RAY_MIN_AGENTS=10000 (agents/system.py, raised in S94):
    # on a single VPS the sequential + asyncio.to_thread path (S65) handles
    # 1000+ agents, while 1000 Ray actors (~150-250 MB each) would OOM.
    agent_count: int = 300

    model_config = {"env_prefix": "EARTHLINK_"}


settings = Settings()
