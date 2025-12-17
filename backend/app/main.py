# backend/app/main.py

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import get_settings
from .schemas import ChatRequest, ChatResponse
from .agent.state_graph import run_agent
from .db import get_engine
from .vdb.qdrant_client import _qdrant_client  # type: ignore

settings = get_settings()

app = FastAPI(title="Fraud Q&A Chatbot")

# CORS (allow localhost frontends; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    # Simple health checks
    db_ok = False
    qdrant_ok = False

    try:
        engine = get_engine()
        with engine.connect() as conn:
            # SQLAlchemy 2.x: pass a text() object
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    try:
        _ = _qdrant_client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok" if db_ok and qdrant_ok else "degraded",
        "db_ok": db_ok,
        "qdrant_ok": qdrant_ok,
        "model": settings.openai_model_name,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history_serialised: List[Dict[str, str]] = []
    if request.history:
        for m in request.history:
            history_serialised.append(
                {
                    "role": m.role,
                    "content": m.content,
                }
            )

    state = run_agent(
        question=request.question,
        history=history_serialised,
    )

    answer = state.get("answer") or ""
    answer_type = state.get("answer_type") or "unknown"
    quality = float(state.get("quality_score") or 0.0)
    sql = state.get("generated_sql")

    sources: List[Dict[str, Any]] = []

    if answer_type == "data":
        rows = state.get("sql_result_rows") or []
        sources.append(
            {
                "type": "sql_result",
                "rows_preview": rows,
            }
        )
    elif answer_type == "document":
        chunks = state.get("context_chunks") or []
        doc_sources: List[Dict[str, Any]] = []
        for c in chunks[:5]:
            payload = c["payload"]
            doc_sources.append(
                {
                    "section": payload.get("section"),
                    "subsection": payload.get("subsection"),
                    "snippet": payload["text"],
                    "rerank_score": c.get("rerank_score"),
                }
            )
        sources.append(
            {
                "type": "document_chunks",
                "chunks": doc_sources,
            }
        )

    return ChatResponse(
        answer=answer,
        answer_type=answer_type,
        quality_score=quality,
        sql=sql,
        sources=sources or None,
    )