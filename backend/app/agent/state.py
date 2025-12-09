from typing import List, Dict, Any, Optional, Literal

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    history: List[Dict[str, str]]
    route: Optional[Literal["data", "document", "none"]]

    generated_sql: Optional[str]
    sql_result_rows: Optional[List[Dict[str, Any]]]
    sql_result_preview: Optional[str]

    context_chunks: Optional[List[Dict[str, Any]]]

    answer: Optional[str]
    answer_type: Optional[str]
    quality_score: Optional[float]