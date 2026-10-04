from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] | None = Field(default=None, max_length=12)
    document_id: Literal["bhatla", "eba"] | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    answer: str = Field(max_length=16000)
    answer_type: Literal["data", "document", "mixed", "other"]
    quality_score: float = Field(ge=0, le=1)
    sql: str | None = None
    sources: list[dict[str, Any]] | None = None
    status: Literal["answered", "clarification", "insufficient_evidence"] = "answered"
    quality_available: bool = False
    quality_method: str = "unavailable"
    quality_breakdown: dict[str, Any] | None = None
    request_id: str = ""
    elapsed_ms: float = 0
    truncated: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
