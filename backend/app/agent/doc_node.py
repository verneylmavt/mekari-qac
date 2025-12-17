# backend/app/agent/doc_nodes.py

from typing import List, Dict, Any

from .state import AgentState
from ..vdb.qdrant_client import retrieve_relevant_chunks
from ..llm.openai_client import call_gpt5_mini


def retrieval_node(state: AgentState) -> AgentState:
    question = state["question"]
    print("Retrieving Document from Vector Database...")
    results = retrieve_relevant_chunks(question, top_k=10, use_reranker=True)
    state["context_chunks"] = results  # type: ignore
    return state


DOC_ANSWER_SYSTEM = (
    "You are answering questions strictly based on the document "
    "“Understanding Credit Card Frauds”.\n\n"
    "You will be given several relevant excerpts from the document. "
    "Using only these excerpts (do not invent facts), answer the question.\n"
    "If the excerpts do not contain enough information to fully answer, "
    "say so clearly."
)


def _build_context(
    chunks: List[Dict[str, Any]],
    max_chars_per_chunk: int = 500,
) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        payload = c["payload"]
        section = payload.get("section")
        subsection = payload.get("subsection")
        text = payload["text"].replace("\n", " ")
        snippet = text[:max_chars_per_chunk]
        parts.append(
            f"[{i}] Section: {section} | Subsection: {subsection}\n{snippet}\n"
        )
    return "\n".join(parts)


def rag_answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    chunks = state.get("context_chunks") or []

    context = _build_context(chunks)

    user_prompt = (
        f"Excerpts from the document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in clear language and refer implicitly to the document's content."
    )

    answer = call_gpt5_mini(
        system_prompt=DOC_ANSWER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=512,
    )
    print("Reasoning for Retrieved Document (by GPT-5 Mini) is Called")
    
    state["answer"] = answer  # type: ignore
    state["answer_type"] = "document"  # type: ignore
    return state