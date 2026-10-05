"""Offline HTTP boundary tests. No request reaches a real service."""

import math
from unittest.mock import Mock

import pytest
import requests

from frontend import app


def response(body, status=200):
    result = Mock(status_code=status)
    result.json.return_value = body
    return result


def test_chat_bounds_network_wait_and_passes_document_filter(monkeypatch):
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return response(
            {"answer": "Grounded answer", "answer_type": "document", "quality_score": 0.8}
        )

    monkeypatch.setattr(requests, "post", post)
    result = app.call_chat("Explain fraud", [], "http://example.test/", document_id="eba")
    assert result["answer"] == "Grounded answer"
    assert sent == [
        (
            "http://example.test/chat",
            {
                "json": {"question": "Explain fraud", "history": [], "document_id": "eba"},
                "timeout": (5.0, 125.0),
            },
        )
    ]


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1, True, "bad", 10**400])
def test_nonfinite_or_invalid_quality_is_unavailable(monkeypatch, score):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: response(
            {"answer": "Answer", "answer_type": "other", "quality_score": score}
        ),
    )
    result = app.call_chat("Question", [], "http://example.test")
    assert result["quality_score"] is None
    assert result["quality_available"] is False


@pytest.mark.parametrize(
    "body", [[], {}, {"answer": 42}, {"answer": ""}, {"answer": "OK", "sources": "invalid"}]
)
def test_malformed_response_is_a_safe_client_error(monkeypatch, body):
    monkeypatch.setattr(requests, "post", lambda *a, **k: response(body))
    with pytest.raises(app.BackendError, match="valid answer"):
        app.call_chat("Question", [], "http://example.test")


def test_structured_error_preserves_safe_retry_and_request_id(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: response(
            {
                "detail": {
                    "code": "busy",
                    "message": "Service is busy.",
                    "retryable": True,
                    "request_id": "abc",
                }
            },
            503,
        ),
    )
    with pytest.raises(app.BackendError) as exc:
        app.call_chat("Question", [], "http://example.test")
    assert str(exc.value) == "Service is busy."
    assert exc.value.retryable is True
    assert exc.value.request_id == "abc"


def test_unstructured_error_never_displays_backend_traceback(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: response({"detail": "password=secret traceback"}, 500)
    )
    with pytest.raises(app.BackendError) as exc:
        app.call_chat("Question", [], "http://example.test")
    assert "secret" not in str(exc.value)
    assert exc.value.retryable is True


def test_timeout_is_explicit_and_never_automatically_retried(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(args)
        raise requests.Timeout("private details")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(app.BackendError, match="timed out") as exc:
        app.call_chat("Question", [], "http://example.test", timeout_s=40)
    assert len(calls) == 1
    assert exc.value.retryable


def test_ready_degraded_retains_metadata(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: response({"status": "degraded", "db_ok": False, "qdrant_ok": True}, 503),
    )
    assert app.call_health("http://example.test")["status"] == "degraded"


def test_ready_falls_back_only_for_legacy_missing_endpoint(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return (
            response({"detail": "Not Found"}, 404)
            if url.endswith("ready")
            else response({"status": "ok", "db_ok": True})
        )

    monkeypatch.setattr(requests, "get", get)
    assert app.call_health("http://example.test")["status"] == "ok"
    assert calls == [
        ("http://example.test/ready", {"timeout": (5.0, 10.0)}),
        ("http://example.test/health", {"timeout": (5.0, 10.0)}),
    ]


def test_non_json_and_network_failure_are_safe_client_errors(monkeypatch):
    invalid = response(None)
    invalid.json.side_effect = ValueError("private parser traceback")
    monkeypatch.setattr(requests, "post", lambda *a, **k: invalid)
    with pytest.raises(app.BackendError, match="valid answer"):
        app.call_chat("Question", [], "http://example.test")

    def disconnected(*args, **kwargs):
        raise requests.ConnectionError("password=secret")

    monkeypatch.setattr(requests, "get", disconnected)
    with pytest.raises(app.BackendError, match="Could not reach") as exc:
        app.call_health("http://example.test")
    assert "secret" not in str(exc.value)


def test_ready_does_not_hide_server_failure_behind_legacy_health(monkeypatch):
    urls = []

    def get(url, **kwargs):
        urls.append(url)
        return response(
            {
                "detail": {
                    "code": "dependency_unavailable",
                    "message": "Documents unavailable.",
                    "retryable": True,
                }
            },
            503,
        )

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(app.BackendError, match="Documents unavailable"):
        app.call_health("http://example.test")
    assert urls == ["http://example.test/ready"]
