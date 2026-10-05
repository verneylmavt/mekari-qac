import time
from functools import lru_cache

from langgraph.graph import END, StateGraph

from ..runtime import Deadline
from ..schemas import ChatResponse
from .data_node import run_sql_node
from .doc_node import rag_answer_node, retrieval_node
from .route_node import router_node
from .score_node import scoring_node
from .state import AgentState


def timed(name, function):
    def node(state):
        state["deadline"].check()
        start = time.monotonic()
        result = function(state)
        result["timings"] = {
            **state.get("timings", {}),
            name: round((time.monotonic() - start) * 1000, 2),
        }
        return result

    return node


@lru_cache
def get_graph():
    graph = StateGraph(AgentState)
    for name, node in (
        ("planner", router_node),
        ("sql", run_sql_node),
        ("retrieval", retrieval_node),
        ("answer", rag_answer_node),
        ("quality", scoring_node),
    ):
        graph.add_node(name, timed(name, node))
    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        lambda s: "quality"
        if s.get("answer")
        else ("sql" if s["plan"].route in ("data", "mixed") else "retrieval"),
        {"quality": "quality", "sql": "sql", "retrieval": "retrieval"},
    )
    graph.add_conditional_edges(
        "sql",
        lambda s: "retrieval" if s["plan"].route == "mixed" else "answer",
        {"retrieval": "retrieval", "answer": "answer"},
    )
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", "quality")
    graph.add_edge("quality", END)
    return graph.compile()


def run_agent(request, resources, deadline=None):
    start = time.monotonic()
    deadline = deadline or Deadline(resources.settings.request_timeout_seconds)
    state = get_graph().invoke(
        {"request": request, "resources": resources, "deadline": deadline, "timings": {}}
    )
    deadline.check()
    sources = []
    if state.get("sql"):
        sources.append(
            {
                "type": "sql_result",
                "citation_id": "SQL",
                "rows_preview": state.get("rows", []),
                "truncated": state.get("truncated", False),
                "scope": "Synthetic transaction warehouse",
            }
        )
    if state.get("chunks"):
        sources.append(
            {
                "type": "document_chunks",
                "chunks": [
                    {
                        **c["payload"],
                        "citation_id": c["citation_id"],
                        "snippet": c["payload"]["text"],
                        "rerank_score": c.get("rerank_score"),
                    }
                    for c in state["chunks"]
                ],
            }
        )
    return ChatResponse(
        answer=state["answer"],
        answer_type=state["answer_type"],
        quality_score=state.get("quality_score", 0),
        sql=state.get("sql"),
        sources=sources or None,
        status=state.get("status", "answered"),
        quality_available=state.get("quality_available", False),
        quality_method=state.get("quality_method", "unavailable"),
        quality_breakdown=state.get("quality_breakdown"),
        truncated=state.get("truncated", False),
        elapsed_ms=round((time.monotonic() - start) * 1000, 2),
        timings=state["timings"],
    )
