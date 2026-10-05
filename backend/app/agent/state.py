from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Plan(StrictOutput):
    route: Literal["data", "document", "mixed", "none"]
    standalone_question: str = Field(min_length=1, max_length=4000)
    document_id: Literal["bhatla", "eba"] | None
    data_question: str = Field(max_length=4000)
    document_question: str = Field(max_length=4000)
    clarification: str = Field(max_length=500)


class SQLDraft(StrictOutput):
    sql: str = Field(min_length=1, max_length=20000)


class AnswerDraft(StrictOutput):
    answer: str = Field(min_length=1, max_length=8000)
    status: Literal["answered", "insufficient_evidence"]
    citations: list[str] = Field(max_length=20)


class Quality(StrictOutput):
    evidence_support: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    consistency: float = Field(ge=0, le=1)
    explanation: str = Field(max_length=1000)


class AgentState(TypedDict, total=False):
    request: Any
    resources: Any
    deadline: Any
    plan: Plan
    rows: list[dict]
    sql: str
    truncated: bool
    chunks: list[dict]
    evidence: str
    answer: str
    answer_type: str
    status: str
    quality_score: float
    quality_available: bool
    quality_method: str
    quality_breakdown: dict
    timings: dict[str, float]
