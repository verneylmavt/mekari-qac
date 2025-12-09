from typing import List, Dict, Any

from .state import AgentState
from ..llm.client import call_gpt5_nano

SCORE_SYSTEM = (
    "You are grading an answer.\n"
    "Given a question, an answer, and some evidence, you must output a single number "
    "between 0.0 and 1.0 (inclusive) representing how correct and well-supported the answer is.\n"
    "0.0 means completely incorrect or unsupported. 1.0 means fully correct and well supported.\n"
    "Return only the number, nothing else."
)


def scoring_node(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state.get("answer") or ""

    # Heuristic base score
    base_score = 0.5

    if state.get("answer_type") == "data":
        rows = state.get("sql_result_rows") or []
        preview = state.get("sql_result_preview") or ""
        if rows and "Error executing SQL" not in preview:
            base_score = 0.75
        elif not rows:
            base_score = 0.3
        evidence = preview
    else:
        chunks: List[Dict[str, Any]] = state.get("context_chunks") or []
        if chunks:
            avg_rerank = sum(c.get("rerank_score", 0.0) for c in chunks) / len(chunks)
            # crude sigmoid mapping to [0,1]
            norm = 1 / (1 + pow(2.71828, -avg_rerank))
            base_score = 0.6 + 0.3 * norm
        else:
            base_score = 0.3

        snippets: List[str] = []
        for c in chunks[:2]:
            text = c["payload"]["text"].replace("\n", " ")
            snippets.append(text[:300])
        evidence = "\n\n".join(snippets)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Score:"
    )

    try:
        raw = call_gpt5_nano(
            system_prompt=SCORE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=10,
        )
        print("Scoring (by GPT-5 Nano) is Called")
        raw = raw.strip()
        score = float(raw)
        score = max(0.0, min(1.0, score))
    except Exception:
        score = base_score

    final_score = (base_score + score) / 2.0
    state["quality_score"] = final_score  # type: ignore
    return state