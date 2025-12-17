# backend/app/config.py

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic v2 style settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DB
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_user: str = Field("user", env="DB_USER")
    db_password: str = Field("password", env="DB_PASSWORD")
    db_name: str = Field("database", env="DB_NAME")

    # Qdrant
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_collection: str = Field("bhatla_credit_fraud", env="QDRANT_COLLECTION")

    # OpenAI / GPT-5 Nano
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model_name: str = Field("gpt-5-mini", env="OPENAI_MODEL_NAME")

    # Embedding / reranker
    embed_model_name: str = Field("BAAI/bge-base-en-v1.5", env="EMBED_MODEL_NAME")
    reranker_model_name: str = Field("BAAI/bge-reranker-base", env="RERANKER_MODEL_NAME")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:"
            f"{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()