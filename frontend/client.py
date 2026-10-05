"""Bounded, single-attempt HTTP calls with safe public errors."""

import math
from typing import Any

import requests


class BackendError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, request_id: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.request_id = request_id


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _error(response: requests.Response, body: Any) -> BackendError:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        message = " ".join(detail["message"].split())[:500]
        request_id = detail.get("request_id")
        return BackendError(
            message or "The backend could not complete this request.",
            retryable=detail.get("retryable") is True,
            request_id=request_id[:100] if isinstance(request_id, str) else None,
        )
    message = (
        "The backend rejected the request. Check the question and try again."
        if response.status_code == 422
        else "The backend could not complete this request."
    )
    return BackendError(message, retryable=response.status_code in (429, 502, 503, 504, 500))


def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
    try:
        return getattr(requests, method)(url, **kwargs)
    except requests.Timeout as exc:
        raise BackendError(
            "The request timed out. The server may still be finishing it. Retry when ready.",
            retryable=True,
        ) from exc
    except requests.RequestException as exc:
        raise BackendError(
            "Could not reach the backend. Check its URL and readiness.", retryable=True
        ) from exc


def validate_chat_response(body: Any) -> dict[str, Any]:
    invalid = BackendError(
        "The backend did not return a valid answer. Check readiness and retry.", retryable=True
    )
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("answer"), str)
        or not body["answer"].strip()
    ):
        raise invalid
    if body.get("sql") is not None and not isinstance(body["sql"], str):
        raise invalid
    sources = body.get("sources")
    if sources is not None:
        if not isinstance(sources, list) or not all(isinstance(source, dict) for source in sources):
            raise invalid
        for source in sources:
            for key in ("rows_preview", "chunks"):
                items = source.get(key)
                if items is not None and (
                    not isinstance(items, list) or not all(isinstance(item, dict) for item in items)
                ):
                    raise invalid
    result = dict(body)
    score = finite_number(body.get("quality_score"))
    available = body.get("quality_available", score is not None) is True
    result["quality_available"] = available and score is not None and 0 <= score <= 1
    result["quality_score"] = score if result["quality_available"] else None
    if result.get("answer_type") not in ("data", "document", "mixed", "other"):
        result["answer_type"] = "other"
    return result


def request_chat(
    question: str,
    history: list[dict[str, str]],
    base_url: str,
    *,
    document_id: str | None = None,
    timeout_s: float = 125.0,
) -> dict[str, Any]:
    response = _request(
        "post",
        f"{base_url.rstrip('/')}/chat",
        json={"question": question, "history": history, "document_id": document_id},
        timeout=(5.0, timeout_s),
    )
    body = _json(response)
    if response.status_code >= 400:
        raise _error(response, body)
    return validate_chat_response(body)


def request_health(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    response = _request("get", f"{base}/ready", timeout=(5.0, 10.0))
    if response.status_code in (404, 405):
        response = _request("get", f"{base}/health", timeout=(5.0, 10.0))
    body = _json(response)
    if isinstance(body, dict) and body.get("status") in ("ok", "ready", "degraded", "not_ready"):
        return body
    if response.status_code >= 400:
        raise _error(response, body)
    raise BackendError("The backend did not return a valid readiness check.")
