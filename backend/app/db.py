from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import Settings, get_settings


def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.max_concurrent_chats,
        max_overflow=0,
        pool_timeout=5,
        connect_args={"connect_timeout": 5},
    )


@lru_cache
def get_engine() -> Engine:
    return create_database_engine(get_settings())
