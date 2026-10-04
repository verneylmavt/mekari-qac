"""Validated configuration. Importing the application never contacts a service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    db_host: str = "localhost"
    db_port: int = Field(5432, ge=1, le=65535)
    db_user: str = "fraud_reader"
    db_password: SecretStr = SecretStr("")
    db_name: str = "database"
    db_admin_user: str = "postgres"
    db_admin_password: SecretStr = SecretStr("")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fraud_documents"
    corpus_dir: Path = ROOT / "data" / "corpus"
    openai_api_key: SecretStr = SecretStr("")
    openai_model_name: str = "gpt-5-mini"
    openai_sql_model: str = "gpt-5-mini"
    openai_planner_model: str = "gpt-5-nano"
    openai_judge_model: str = "gpt-5-nano"
    embed_model_name: str = "BAAI/bge-base-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    max_concurrent_chats: int = Field(4, ge=1, le=32)
    max_concurrent_inference: int = Field(1, ge=1, le=4)
    request_timeout_seconds: float = Field(120, gt=0, le=600)
    external_timeout_seconds: float = Field(30, gt=0, le=120)
    sql_timeout_ms: int = Field(10000, ge=1, le=10000)
    sql_max_rows: int = Field(200, ge=1, le=200)
    sql_preview_rows: int = Field(20, ge=1, le=50)
    retrieval_candidates: int = Field(30, ge=1, le=100)
    retrieval_final_chunks: int = Field(8, ge=1, le=20)
    document_context_tokens: int = Field(6000, ge=100, le=10000)
    retrieval_cache_size: int = Field(128, ge=0, le=1000)

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg2",
            username=self.db_user,
            password=self.db_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
