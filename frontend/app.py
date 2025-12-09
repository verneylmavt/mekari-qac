# frontend/streamlit_app.py

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


# --------- Config helpers --------- #

def get_default_backend_url() -> str:
    env_url = os.getenv("FRAUD_API_BASE_URL", "http://localhost:8000")
    return env_url.rstrip("/")


def get_backend_url() -> str:
    # Backed by Streamlit session_state but with env-based default
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = get_default_backend_url()
    return st.session_state.backend_url.rstrip("/")


# --------- Backend client --------- #

def call_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    base = (base_url or get_backend_url()).rstrip("/")
    url = f"{base}/health"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# def call_chat(
#     question: str,
#     history: List[Dict[str, str]],
#     base_url: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     history: list of {"role": "user"|"assistant", "content": "..."}.
#     """
#     base = (base_url or get_backend_url()).rstrip("/")
#     url = f"{base}/chat"

#     payload: Dict[str, Any] = {
#         "question": question,
#         "history": history,
#     }

#     resp = requests.post(url, json=payload, timeout=30)
#     resp.raise_for_status()
#     return resp.json()

def call_chat(
    question: str,
    history: List[Dict[str, str]],
    base_url: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    history: list of {"role": "user"|"assistant", "content": "..."}.
    timeout_s: HTTP timeout in seconds for the /chat request.
    """
    base = (base_url or get_backend_url()).rstrip("/")
    url = f"{base}/chat"

    payload: Dict[str, Any] = {
        "question": question,
        "history": history,
    }

    # # default to a more realistic timeout for multiple LLM + DB calls
    # if timeout_s is None:
    #     timeout_s = 120.0  # 2 minutes

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


# --------- UI helpers --------- #

def init_session_state() -> None:
    if "messages" not in st.session_state:
        # Each message: {"role": "user"/"assistant", "content": "...", ...meta}
        st.session_state.messages: List[Dict[str, Any]] = []

    if "last_health" not in st.session_state:
        st.session_state.last_health = None


def build_history_for_backend(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Map our rich messages down to the {role, content} list expected by the backend.
    We intentionally ignore metadata (answer_type, quality, etc.).
    """
    history: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def render_sources(sources: Optional[List[Dict[str, Any]]]) -> None:
    if not sources:
        return

    for src in sources:
        src_type = src.get("type")
        if src_type == "sql_result":
            rows_preview = src.get("rows_preview") or []
            with st.expander("Preview of Queried Data"):
                # st.write(
                #     "This is a preview of the rows returned by the SQL query used for this answer."
                # )
                st.json(rows_preview)
        elif src_type == "document_chunks":
            chunks = src.get("chunks") or []
            with st.expander("Preview of Retrieved Document"):
                for i, c in enumerate(chunks, start=1):
                    section = c.get("section") or "-"
                    subsection = c.get("subsection") or "-"
                    snippet = c.get("snippet") or ""
                    rerank_score = c.get("rerank_score", None)

                    st.markdown(f"**Chunk {i}**")
                    st.markdown(f"**Section:** {section}")
                    if subsection and subsection != "-":
                        st.markdown(f"_Subsection_: {subsection}")
                    if rerank_score is not None:
                        try:
                            st.caption(f"Rerank score: {float(rerank_score):.3f}")
                        except Exception:
                            st.caption(f"Rerank score: {rerank_score}")
                    st.write(snippet)
                    st.markdown("---")


def render_assistant_message(msg: Dict[str, Any]) -> None:
    content = msg.get("content", "")
    answer_type = msg.get("answer_type", "unknown")
    quality_score = msg.get("quality_score", None)
    sql = msg.get("sql")
    sources = msg.get("sources")

    st.markdown(content)

    # Meta information
    meta_parts: List[str] = []
    meta_parts.append(f"Source: `{answer_type}`")
    if quality_score is not None:
        try:
            meta_parts.append(f"LLM-as-Judge Score: {float(quality_score):.3f}")
        except Exception:
            meta_parts.append(f"LLM-as-Judge Score: {quality_score}")
    # if sql and answer_type == "data":
    #     meta_parts.append("SQL-backed")

    if meta_parts:
        st.caption(" • ".join(meta_parts))

    # Show SQL if available
    if sql:
        with st.expander("SQL Query"):
            st.code(sql, language="sql")

    # Show sources if available
    render_sources(sources)


def sidebar_layout() -> None:
    st.sidebar.header("Backend")

    backend_url = get_backend_url()
    new_backend_url = st.sidebar.text_input(
        "Backend URL:",
        value=backend_url,
        help="FastAPI URL (e.g. http://localhost:8000)",
    )
    st.session_state.backend_url = new_backend_url.rstrip("/")

    # Health check
    if st.sidebar.button("Check Health"):
        try:
            health = call_health()
            st.session_state.last_health = health
        except Exception as e:
            st.sidebar.error(f"Health check failed: {e}")
            st.session_state.last_health = None

    health = st.session_state.last_health
    if health is not None:
        status = health.get("status", "unknown")
        db_ok = health.get("db_ok", False)
        qdrant_ok = health.get("qdrant_ok", False)
        model = health.get("model", "unknown")

        # st.sidebar.subheader("Last health check")
        st.sidebar.markdown("---")
        st.sidebar.write(f"Status: `{status}`")
        st.sidebar.write(f"PostgreSQL Relational Database: `{db_ok}`")
        st.sidebar.write(f"Qdrant Vector Database: `{qdrant_ok}`")
        st.sidebar.write(f"LLM Model: `{model}`")

    # if st.sidebar.button("Clear Conversation"):
    #     st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="Fraud Q&A Chatbot",
        page_icon="💳",
        layout="centered",
    )

    init_session_state()
    sidebar_layout()

    st.title("Fraud Q&A Chatbot")
    st.write(
        """
        Ask me any questions about 
        [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraud%20dataset)
        or [Understanding Credit Card Frauds by TATA Consultancy](https://popcenter.asu.edu/sites/g/files/litvpz3631/files/problems/credit_card_fraud/PDFs/Bhatla.pdf).
        I'm happy to help! 😄
        """
    )
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []

    # Render existing chat history
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                render_assistant_message(msg)

    # Chat input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # 1. Append user message to local history for display
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)

        # Display it immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Build history for backend (exclude the current assistant reply)
        #    We send only messages BEFORE this last user turn
        history_for_backend = build_history_for_backend(
            st.session_state.messages[:-1]
        )

        # 3. Call backend /chat
        try:
            response = call_chat(
                question=user_input,
                history=history_for_backend,
            )
            answer = response.get("answer", "")
            answer_type = response.get("answer_type", "unknown")
            quality_score = response.get("quality_score", 0.0)
            sql = response.get("sql")
            sources = response.get("sources")

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": answer,
                "answer_type": answer_type,
                "quality_score": quality_score,
                "sql": sql,
                "sources": sources,
            }
            st.session_state.messages.append(assistant_msg)

            # Render assistant message
            with st.chat_message("assistant"):
                render_assistant_message(assistant_msg)

        except requests.HTTPError as http_err:
            error_msg = f"Backend HTTP error: {http_err}"
            st.error(error_msg)
            # Append an error-style assistant message for consistency
            assistant_msg = {
                "role": "assistant",
                "content": error_msg,
                "answer_type": "error",
                "quality_score": 0.0,
                "sql": None,
                "sources": None,
            }
            st.session_state.messages.append(assistant_msg)
        except Exception as e:
            error_msg = f"Backend request failed: {e}"
            st.error(error_msg)
            assistant_msg = {
                "role": "assistant",
                "content": error_msg,
                "answer_type": "error",
                "quality_score": 0.0,
                "sql": None,
                "sources": None,
            }
            st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()