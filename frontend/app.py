"""Streamlit fraud workbench. Run from the repository root."""

import os
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from frontend.client import BackendError, finite_number, request_chat, request_health
from frontend.presentation import (
    begin_turn,
    chart_spec,
    citation_label,
    export_markdown,
    retry_turn,
)
from frontend.presentation import (
    build_history_for_backend as build_history_for_backend,
)

DOCUMENTS = {
    "Auto · all documents": None,
    "Bhatla · credit card fraud": "bhatla",
    "EBA · payment fraud 2024": "eba",
}
STARTERS = [
    ("Monthly trend", "What is the monthly fraud rate in the transaction dataset?"),
    ("Fraud mechanisms", "What credit card fraud mechanisms does Bhatla describe?"),
    ("Payment fraud", "What does the EBA 2024 report say about payment fraud?"),
    (
        "Connect the evidence",
        "Which transaction categories have the highest fraud rate, and what fraud prevention measures do the documents discuss?",
    ),
]


def get_default_backend_url() -> str:
    return os.getenv("FRAUD_API_BASE_URL", "http://localhost:8000").rstrip("/")


def get_backend_url() -> str:
    return st.session_state.get("backend_url", get_default_backend_url()).strip().rstrip("/")


def call_chat(
    question: str,
    history: list[dict[str, str]],
    base_url: str | None = None,
    timeout_s: float | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    return request_chat(
        question,
        history,
        base_url or get_backend_url(),
        document_id=document_id,
        timeout_s=125.0 if timeout_s is None else timeout_s,
    )


def call_health(base_url: str | None = None) -> dict[str, Any]:
    return request_health(base_url or get_backend_url())


def init_session_state() -> None:
    for key, default in {
        "messages": [],
        "turns": [],
        "last_health": None,
        "health_checked_at": None,
        "backend_url": get_default_backend_url(),
        "health_base_url": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _pending() -> bool:
    return any(turn.get("state") == "pending" for turn in st.session_state.turns)


def _submit(question: str | None = None) -> None:
    if _pending():
        return
    question = question or st.session_state.get("chat_question", "")
    if not question.strip():
        return
    if len(question.strip()) > 4000:
        st.session_state.input_error = "Keep your question within 4,000 characters."
        return
    turn = begin_turn(
        st.session_state.messages,
        question,
        DOCUMENTS[st.session_state.get("document_scope", "Auto · all documents")],
    )
    st.session_state.turns.append(turn)
    st.session_state.submit_once = turn["id"]
    st.session_state.pop("input_error", None)


def _retry(turn_id: str) -> None:
    if _pending():
        return
    for index, turn in enumerate(st.session_state.turns):
        if turn["id"] == turn_id and turn["state"] == "failed" and turn.get("retryable"):
            st.session_state.turns[index] = retry_turn(st.session_state.messages, turn)
            st.session_state.submit_once = turn_id
            return


def _clear() -> None:
    if not _pending():
        st.session_state.messages = []
        st.session_state.turns = []


def sidebar_layout(pending: bool) -> None:
    with st.sidebar:
        st.markdown("### Connection")
        st.text_input("Backend URL", key="backend_url", disabled=pending)
        base = get_backend_url()
        if base != st.session_state.health_base_url:
            st.session_state.last_health = None
            st.session_state.health_checked_at = None
            st.session_state.health_base_url = base
        if st.button("Check readiness", disabled=pending, use_container_width=True):
            try:
                st.session_state.last_health = call_health(base)
            except BackendError as error:
                st.session_state.last_health = {"status": "unreachable", "message": str(error)}
            st.session_state.health_checked_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        health = st.session_state.last_health
        if health:
            status = health.get("status", "unknown")
            if status in ("ok", "ready"):
                st.success("Backend ready")
            else:
                st.warning(f"Backend {str(status).replace('_', ' ')}")
            st.caption(f"Checked {st.session_state.health_checked_at}")
            if health.get("message"):
                st.caption(health["message"])
            for key, label in [
                ("db_ok", "Warehouse"),
                ("qdrant_ok", "Document retrieval"),
                ("openai_configured", "Answer model configured"),
            ]:
                if key in health:
                    st.caption(f"{label}: {'available' if health[key] else 'unavailable'}")
        else:
            st.caption("Readiness has not been checked for this URL.")
        st.divider()
        st.caption(
            "Bounded requests · up to 200 SQL rows · evidence previews may be capped. Actual result limits appear with each answer."
        )


def render_sources(sources: list[dict[str, Any]] | None) -> None:
    for source in sources or []:
        if source.get("type") == "sql_result":
            rows = source.get("rows_preview") or []
            with st.expander(f"Transaction evidence · {len(rows)} preview rows"):
                if source.get("truncated"):
                    st.warning("Result limit reached. This preview is incomplete.")
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("The query returned no rows.")
        elif source.get("type") == "document_chunks":
            chunks = source.get("chunks") or []
            with st.expander(f"Document evidence · {len(chunks)} references"):
                for index, chunk in enumerate(chunks, 1):
                    payload = (
                        chunk.get("payload") if isinstance(chunk.get("payload"), dict) else chunk
                    )
                    st.markdown(f"**{citation_label(chunk, index)}**")
                    st.write(
                        payload.get("text") or payload.get("snippet") or "No excerpt returned."
                    )
                    score = finite_number(chunk.get("rerank_score"))
                    if score is not None:
                        st.caption(
                            f"Retrieval relevance: {score:.3f} · retrieval scores are not answer confidence"
                        )
                    if index < len(chunks):
                        st.divider()


def render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(message.get("content", ""))
    answer_type = message.get("answer_type", "other")
    status = message.get("status", "answered")
    score = finite_number(message.get("quality_score"))
    quality = (
        f"{score:.0%}"
        if message.get("quality_available", score is not None) and score is not None
        else "N/A"
    )
    meta = [str(answer_type).capitalize(), f"Evidence quality: {quality}"]
    elapsed = finite_number(message.get("elapsed_ms"))
    if elapsed is not None and elapsed >= 0:
        meta.append(f"{elapsed / 1000:.1f}s")
    st.caption(" · ".join(meta))
    st.caption(
        "Quality is a rubric score for evidence grounding and completeness, not a probability of correctness."
    )
    if status == "clarification":
        st.info("Add the requested detail in your next question.")
    elif status == "insufficient_evidence":
        st.warning("The available evidence does not support a complete answer.")
    if message.get("truncated"):
        st.warning("Some evidence was capped. Interpret this answer within the returned sample.")
    chart = chart_spec(message.get("sources"))
    if chart:
        st.caption(f"{chart['y']} · returned SQL evidence")
        renderer = st.line_chart if chart["kind"] == "line" else st.bar_chart
        renderer(
            chart["rows"], x=chart["x"], y=chart["y"], color="#0f766e", use_container_width=True
        )
    if message.get("sql"):
        with st.expander("SQL query"):
            st.code(message["sql"], language="sql")
    render_sources(message.get("sources"))
    with st.expander("Quality and request details"):
        if quality == "N/A":
            st.caption("Quality assessment was unavailable. No score is shown.")
        breakdown = message.get("quality_breakdown")
        if isinstance(breakdown, dict) and breakdown:
            st.json(breakdown)
        if message.get("quality_method"):
            st.caption(f"Method: {message['quality_method']}")
        if message.get("request_id"):
            st.caption(f"Request: {message['request_id']}")
        if isinstance(message.get("timings"), dict) and message["timings"]:
            st.json(message["timings"])


def _finish_failure(turn: dict[str, Any], error: BackendError) -> None:
    turn.update(
        state="failed", error=str(error), retryable=error.retryable, request_id=error.request_id
    )
    for message in st.session_state.messages:
        if message.get("turn_id") == turn["id"]:
            message["state"] = "failed"


def _process_submission(turn_id: str) -> None:
    turn = next(
        (
            turn
            for turn in st.session_state.turns
            if turn["id"] == turn_id and turn["state"] == "pending"
        ),
        None,
    )
    if turn is None:
        return
    with st.chat_message("assistant", avatar=":material/insights:"):
        with st.spinner("Checking the question and gathering evidence…"):
            try:
                response = call_chat(
                    turn["question"], turn["history"], document_id=turn["document_id"]
                )
            except BackendError as error:
                _finish_failure(turn, error)
            else:
                turn["state"] = "completed"
                for message in st.session_state.messages:
                    if message.get("turn_id") == turn["id"]:
                        message["state"] = "completed"
                answer = dict(
                    response,
                    role="assistant",
                    content=response["answer"],
                    turn_id=turn["id"],
                    state="completed",
                )
                user_index = next(
                    index
                    for index, message in enumerate(st.session_state.messages)
                    if message.get("turn_id") == turn["id"] and message.get("role") == "user"
                )
                st.session_state.messages.insert(user_index + 1, answer)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Fraud Intelligence | Mekari", page_icon="◈", layout="centered")
    init_session_state()
    submit_once = st.session_state.pop("submit_once", None)
    # An interrupted request requires explicit retry; a rerun must never replay POST.
    if submit_once is None:
        for turn in st.session_state.turns:
            if turn["state"] == "pending":
                _finish_failure(
                    turn,
                    BackendError(
                        "The previous request was interrupted. It may still be finishing on the server.",
                        retryable=True,
                    ),
                )
    pending = _pending()
    st.markdown(
        """<style>
    .stApp {background:#f7f9fb; color:#172b3a;}
    [data-testid="stSidebar"] {background:#edf3f5;}
    .block-container {max-width:960px; padding-top:2.7rem; padding-bottom:5rem;}
    h1 {letter-spacing:-.045em; color:#172b3a;}
    [data-testid="stChatMessage"] {background:white; border:1px solid #dfe8ec; border-radius:14px; padding:1.2rem;}
    .stButton button, .stDownloadButton button {border-radius:8px; border-color:#cbdadd;}
    .stButton button:hover, .stDownloadButton button:hover {border-color:#0f766e; color:#0f766e;}
    @media(max-width:640px) {.block-container {padding-top:1.4rem; padding-left:1rem; padding-right:1rem;}}
    </style>""",
        unsafe_allow_html=True,
    )
    sidebar_layout(pending)
    if os.getenv("FRAUD_DEMO_MODE") == "1":
        st.info("Offline demo · fixture answers and sample transaction rows. No live model calls.")
    st.caption("MEKARI  /  FRAUD INTELLIGENCE")
    st.title("Follow the evidence.")
    st.write(
        "Explore transaction patterns, understand fraud mechanisms, and connect each answer to its sources."
    )
    st.caption("SYNTHETIC TRANSACTIONS  ·  BHATLA REPORT  ·  EBA 2024 REPORT")
    st.selectbox(
        "Document scope",
        list(DOCUMENTS),
        key="document_scope",
        disabled=pending,
        help="Filters document evidence. Transaction questions still use the warehouse.",
    )
    st.caption(
        "Transaction data is synthetic. Document findings describe their original populations and reporting periods."
    )
    st.divider()
    if not st.session_state.messages:
        st.markdown("#### Start with a question")
        columns = st.columns(2)
        for index, (label, question) in enumerate(STARTERS):
            with columns[index % 2]:
                st.button(
                    label,
                    key=f"starter_{index}",
                    help=question,
                    use_container_width=True,
                    disabled=pending,
                    on_click=_submit,
                    args=(question,),
                )
        st.caption(
            "Answers include source references, SQL evidence when used, and transparent quality details."
        )
    else:
        tools = st.columns([1, 1, 2])
        with tools[0]:
            st.button(
                "Clear conversation", disabled=pending, on_click=_clear, use_container_width=True
            )
        with tools[1]:
            st.download_button(
                "Export Markdown",
                export_markdown(st.session_state.messages),
                file_name="fraud-conversation.md",
                mime="text/markdown",
                disabled=pending,
                use_container_width=True,
            )
        with tools[2]:
            st.caption(
                f"{sum(turn['state'] == 'completed' for turn in st.session_state.turns)} completed questions"
            )
    for message in st.session_state.messages:
        with st.chat_message(
            message.get("role", "assistant"),
            avatar=":material/insights:" if message.get("role") == "assistant" else None,
        ):
            if message.get("role") == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message.get("content", ""))
                if message.get("state") == "failed":
                    turn = next(
                        (
                            turn
                            for turn in st.session_state.turns
                            if turn["id"] == message.get("turn_id")
                        ),
                        None,
                    )
                    if turn:
                        st.error(turn.get("error", "The request could not be completed."))
                        if turn.get("request_id"):
                            st.caption(f"Request: {turn['request_id']}")
                        if turn.get("retryable"):
                            st.button(
                                "Retry question",
                                key=f"retry_{turn['id']}",
                                disabled=pending,
                                on_click=_retry,
                                args=(turn["id"],),
                            )
    if st.session_state.get("input_error"):
        st.error(st.session_state.input_error)
    st.chat_input(
        "Ask about fraud, transactions, or the reports…",
        key="chat_question",
        disabled=pending,
        max_chars=4000,
        on_submit=_submit,
    )
    if submit_once:
        _process_submission(submit_once)


if __name__ == "__main__":
    main()
