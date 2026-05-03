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

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # WebTransport (QUIC/HTTP3)
    wt_enabled: bool = False
    wt_host: str = "0.0.0.0"
    wt_port: int = 4433
    wt_path: str = "/wt/world"
    wt_cert_path: str = ""
    wt_key_path: str = ""
    # Optional externally reachable WT URL (for reverse proxy / separate hostname).
    # Example: "https://earthlink.yuxilabs.com/wt/world" or "https://wt.earthlink.yuxilabs.com/wt/world"
    wt_public_url: str = ""

    # World — overridable via EARTHLINK_AGENT_COUNT etc.
    # Sequential mode threshold is RAY_MIN_AGENTS=500 (in agents/system.py);
    # values <500 keep sequential tick path, >=500 enable Ray actors.
    agent_count: int = 300

    model_config = {"env_prefix": "EARTHLINK_"}


settings = Settings()
