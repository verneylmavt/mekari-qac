# backend/app/schemas.py

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    answer_type: str
    quality_score: float
    sql: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None