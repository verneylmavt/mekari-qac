# backend/app/agent/graph.py

from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from .state import AgentState
from .router import router_node, fallback_answer_node
from .data_nodes import generate_sql_node, run_sql_node, data_answer_node
from .doc_nodes import retrieval_node, rag_answer_node
from .scoring_node import scoring_node


def _route_decider(state: AgentState) -> str:
    route = state.get("route")
    if route in ("data", "document", "none"):
        return route  # type: ignore
    # Default to none (fallback) if something weird happens
    return "none"


# Build LangGraph
_graph = StateGraph(AgentState)

_graph.add_node("router", router_node)
_graph.add_node("generate_sql", generate_sql_node)
_graph.add_node("run_sql", run_sql_node)
_graph.add_node("data_answer", data_answer_node)
_graph.add_node("retrieve", retrieval_node)
_graph.add_node("rag_answer", rag_answer_node)
_graph.add_node("fallback_answer", fallback_answer_node)
_graph.add_node("score", scoring_node)

_graph.set_entry_point("router")

_graph.add_conditional_edges(
    "router",
    _route_decider,
    {
        "data": "generate_sql",
        "document": "retrieve",
        "none": "fallback_answer",
    },
)

_graph.add_edge("generate_sql", "run_sql")
_graph.add_edge("run_sql", "data_answer")
_graph.add_edge("data_answer", "score")

_graph.add_edge("retrieve", "rag_answer")
_graph.add_edge("rag_answer", "score")

_graph.add_edge("fallback_answer", "score")

_graph.add_edge("score", END)

_app = _graph.compile()


def run_agent(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AgentState:
    initial_state: AgentState = {
        "question": question,
        "history": history or [],
    }
    final_state: AgentState = _app.invoke(initial_state)
    return final_state