"""Streamlit 1.41 AppTest coverage with requests patched at the HTTP boundary."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from frontend import app

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def create_app_test():
    # Chart schema initialization can exceed the framework's 3s default on a cold CI host.
    return AppTest.from_file(APP, default_timeout=10)


def response(body, status=200):
    result = Mock(status_code=status)
    result.json.return_value = body
    return result


@pytest.fixture
def offline(monkeypatch):
    calls = []
    outcomes = []

    def post(url, **kwargs):
        calls.append(kwargs)
        if outcomes:
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return response(
            {
                "answer": "The monthly fraud rate is 1.2% [SQL-1].",
                "answer_type": "data",
                "quality_score": 0.82,
                "quality_available": True,
                "status": "answered",
                "quality_breakdown": {"grounding": 0.9},
                "request_id": "test-id",
                "elapsed_ms": 1500,
                "truncated": False,
                "sql": "SELECT year_month, fraud_rate FROM agg_monthly_fraud",
                "sources": [
                    {
                        "type": "sql_result",
                        "rows_preview": [{"year_month": "2020-01", "fraud_rate": 0.012}],
                    }
                ],
            }
        )

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: response({"status": "ready", "db_ok": True, "qdrant_ok": True}),
    )
    return calls, outcomes


def button(at, label):
    return next(item for item in at.button if item.label == label)


def assert_clean(at):
    assert not at.exception, [exception.message for exception in at.exception]


def test_document_selector_is_visible_in_main_workspace_and_keeps_api_filter(offline):
    calls, _ = offline
    at = create_app_test().run()
    assert_clean(at)
    assert len(at.main.selectbox) == 1
    assert not at.sidebar.selectbox
    selector = at.main.selectbox[0]
    assert selector.key == "document_scope"
    selector.select("EBA · payment fraud 2024").run()
    at.chat_input[0].set_value("What does EBA report?").run()
    assert_clean(at)
    assert calls[0]["json"]["document_id"] == "eba"


def test_starter_submits_once_and_reruns_never_repost(offline):
    calls, _ = offline
    at = create_app_test().run()
    assert_clean(at)
    button(at, "Monthly trend").click().run()
    assert_clean(at)
    assert len(calls) == 1
    assert calls[0]["json"]["history"] == []
    assert calls[0]["json"]["document_id"] is None
    assert len(at.session_state.messages) == 2
    assert at.session_state.turns[0]["state"] == "completed"
    assert "1.2%" in at.chat_message[1].markdown[0].value
    assert at.get("download_button")[0].proto.label == "Export Markdown"
    at.run()
    at.selectbox[0].select("Bhatla · credit card fraud").run()
    assert_clean(at)
    assert len(calls) == 1


def test_submit_sends_previous_history_and_document_filter(offline):
    calls, _ = offline
    at = create_app_test().run()
    at.chat_input[0].set_value("First question").run()
    at.selectbox[0].select("EBA · payment fraud 2024").run()
    at.chat_input[0].set_value("What about EBA?").run()
    assert_clean(at)
    assert len(calls) == 2
    assert calls[1]["json"] == {
        "question": "What about EBA?",
        "document_id": "eba",
        "history": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "The monthly fraud rate is 1.2% [SQL-1]."},
        ],
    }
    button(at, "Clear conversation").click().run()
    assert_clean(at)
    assert at.session_state.messages == []
    assert len(calls) == 2


def test_explicit_retry_preserves_original_history_and_avoids_duplicates(offline):
    calls, outcomes = offline
    at = create_app_test().run()
    at.chat_input[0].set_value("First question").run()
    outcomes.append(requests.Timeout("private"))
    at.chat_input[0].set_value("Failed question").run()
    assert_clean(at)
    failed_id = at.session_state.turns[1]["id"]
    assert len(at.session_state.messages) == 3
    assert "timed out" in at.error[0].value
    at.run()
    assert len(calls) == 2
    at.selectbox[0].select("EBA · payment fraud 2024").run()
    button(at, "Retry question").click().run()
    assert_clean(at)
    assert len(calls) == 3
    assert calls[1]["json"] == calls[2]["json"]
    assert len(at.session_state.messages) == 4
    assert at.session_state.turns[1]["id"] == failed_id
    assert at.session_state.turns[1]["state"] == "completed"


def test_failed_turn_is_absent_from_next_question_history(offline):
    calls, outcomes = offline
    outcomes.append(
        response({"detail": {"code": "busy", "message": "Backend busy.", "retryable": True}}, 503)
    )
    at = create_app_test().run()
    at.chat_input[0].set_value("Failed question").run()
    at.chat_input[0].set_value("Fresh question").run()
    assert_clean(at)
    assert calls[1]["json"]["history"] == []


def test_readiness_timestamp_invalidates_when_backend_url_changes(offline):
    at = create_app_test().run()
    button(at, "Check readiness").click().run()
    assert_clean(at)
    assert at.session_state.health_checked_at.endswith("UTC")
    assert at.success[0].value == "Backend ready"
    at.text_input[0].set_value("http://other.test").run()
    assert_clean(at)
    assert at.session_state.last_health is None
    assert at.session_state.health_checked_at is None


def test_legacy_documents_and_judge_outage_render_without_exception(offline):
    _, outcomes = offline
    outcomes.append(
        response(
            {
                "answer": "Grounded document answer [EBA-1]",
                "answer_type": "document",
                "quality_available": False,
                "quality_score": 0.8,
                "sources": [
                    {
                        "type": "document_chunks",
                        "chunks": [
                            {
                                "citation_id": "EBA-1",
                                "title": "EBA 2024",
                                "page": 7,
                                "snippet": "Document evidence",
                                "rerank_score": 0.8,
                            },
                            {
                                "section": "Legacy section",
                                "subsection": "Legacy subsection",
                                "snippet": "Legacy evidence",
                            },
                        ],
                    }
                ],
            }
        )
    )
    at = create_app_test().run()
    at.chat_input[0].set_value("Explain fraud").run()
    assert_clean(at)
    assert any("Evidence quality: N/A" in caption.value for caption in at.caption)
    assert any("EBA 2024 · page 7" in item.value for item in at.markdown)
    assert any("Legacy evidence" in item.value for item in at.markdown)


def test_interrupted_pending_turn_is_not_replayed(offline):
    calls, _ = offline
    at = create_app_test().run()
    at.session_state.turns = [
        {
            "id": "interrupted",
            "state": "pending",
            "question": "Old request",
            "history": [],
            "document_id": None,
        }
    ]
    at.session_state.messages = [
        {"turn_id": "interrupted", "role": "user", "content": "Old request", "state": "pending"}
    ]
    at.run()
    assert_clean(at)
    assert calls == []
    assert at.session_state.turns[0]["state"] == "failed"
    assert button(at, "Retry question")


def test_retry_earlier_failed_turn_keeps_question_answer_order(offline):
    _, outcomes = offline
    outcomes.append(requests.Timeout())
    at = create_app_test().run()
    at.chat_input[0].set_value("Earlier failed question").run()
    at.chat_input[0].set_value("Later question").run()
    button(at, "Retry question").click().run()
    assert_clean(at)
    assert [(message["role"], message["content"]) for message in at.session_state.messages] == [
        ("user", "Earlier failed question"),
        ("assistant", "The monthly fraud rate is 1.2% [SQL-1]."),
        ("user", "Later question"),
        ("assistant", "The monthly fraud rate is 1.2% [SQL-1]."),
    ]


def test_pending_request_blocks_duplicate_submit_clear_and_retry(monkeypatch, offline):
    observed = []

    def post(url, **kwargs):
        turn_id = st.session_state.turns[0]["id"]
        observed.append(st.session_state.turns[0]["state"])
        app._submit("Duplicate question")
        app._clear()
        app._retry(turn_id)
        assert len(st.session_state.messages) == 1
        assert len(st.session_state.turns) == 1
        return response({"answer": "Answer", "answer_type": "other", "quality_score": 0.5})

    monkeypatch.setattr(requests, "post", post)
    at = create_app_test().run()
    at.chat_input[0].set_value("Question").run()
    assert_clean(at)
    assert observed == ["pending"]
    assert len(at.session_state.messages) == 2
