"""Lifespan-owned boundaries. Heavy models and connections remain lazy."""

import threading
import time

from sqlalchemy import text

from .config import Settings
from .db import create_database_engine
from .runtime import InferenceGate


class Resources:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine = None
        self._llm = None
        self._retriever = None
        self._lock = threading.RLock()
        self.inference_gate = InferenceGate(settings.max_concurrent_inference)
        self._ready_cache = None

    @property
    def engine(self):
        with self._lock:
            if self._engine is None:
                self._engine = create_database_engine(self.settings)
            return self._engine

    @property
    def llm(self):
        with self._lock:
            if self._llm is None:
                from .llm.openai_client import LLMClient

                self._llm = LLMClient(self.settings)
            return self._llm

    @property
    def retriever(self):
        with self._lock:
            if self._retriever is None:
                from .vdb.qdrant_client import DocumentRetriever

                self._retriever = DocumentRetriever(self.settings, self.inference_gate)
            return self._retriever

    def query(self, sql, deadline):
        from .rdb.postgresql_client import run_sql_query

        return run_sql_query(
            sql,
            engine=self.engine,
            max_rows=self.settings.sql_max_rows,
            timeout_ms=max(1, int(deadline.timeout(self.settings.sql_timeout_ms / 1000) * 1000)),
        )

    def ready(self):
        from .rdb.postgresql_client import WAREHOUSE_TABLES

        with self._lock:
            if self._ready_cache and time.monotonic() - self._ready_cache[0] < 5:
                return self._ready_cache[1].copy()
        result = {
            "db_ok": False,
            "qdrant_ok": False,
            "models_cached": False,
            "openai_configured": bool(self.settings.openai_api_key.get_secret_value()),
            "model": self.settings.openai_model_name,
        }
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    connection.execute(text("SET TRANSACTION READ ONLY"))
                    connection.execute(text("SET LOCAL statement_timeout = 3000"))
                    role = connection.execute(
                        text(
                            "SELECT rolsuper, rolcreaterole, rolcreatedb, rolbypassrls "
                            "FROM pg_roles WHERE rolname = current_user"
                        )
                    ).one()
                    objects = all(
                        connection.execute(
                            text("SELECT to_regclass(:name)"), {"name": "public." + name}
                        ).scalar()
                        for name in WAREHOUSE_TABLES
                    )
                    writable = (
                        any(
                            connection.execute(
                                text(
                                    "SELECT has_table_privilege(current_user, :name, 'INSERT,UPDATE,DELETE,TRUNCATE')"
                                ),
                                {"name": "public." + name},
                            ).scalar()
                            for name in WAREHOUSE_TABLES
                        )
                        if objects
                        else True
                    )
                    readable = objects and all(
                        connection.execute(
                            text("SELECT has_table_privilege(current_user, :name, 'SELECT')"),
                            {"name": "public." + name},
                        ).scalar()
                        for name in WAREHOUSE_TABLES
                    )
                    schema_usage = connection.execute(
                        text("SELECT has_schema_privilege(current_user, 'public', 'USAGE')")
                    ).scalar()
                    result["db_ok"] = (
                        objects and readable and schema_usage and not any(role) and not writable
                    )
        except Exception:
            pass
        try:
            readiness = self.retriever.ready()
            if isinstance(readiness, dict):
                result.update(
                    {key: bool(readiness.get(key, False)) for key in ("qdrant_ok", "models_cached")}
                )
            else:
                result["qdrant_ok"] = result["models_cached"] = bool(readiness)
        except Exception:
            pass
        result["status"] = (
            "ok"
            if all(
                result[key] for key in ("db_ok", "qdrant_ok", "models_cached", "openai_configured")
            )
            else "degraded"
        )
        with self._lock:
            self._ready_cache = (time.monotonic(), result.copy())
        return result

    def close(self):
        for resource in (self._llm, self._retriever):
            if resource is not None:
                resource.close()
        if self._engine is not None:
            self._engine.dispose()
