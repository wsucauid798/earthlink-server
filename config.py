from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
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

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "EARTHLINK_"}


settings = Settings()
