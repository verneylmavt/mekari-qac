from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.runtime import ServiceError


class Resources:
    def close(self):
        pass

    def ready(self):
        return {
            "status": "degraded",
            "db_ok": False,
            "qdrant_ok": False,
            "models_cached": False,
            "openai_configured": False,
            "model": "gpt-5-mini",
        }


def test_health_compatible_readiness_strict_liveness_independent():
    with TestClient(create_app(resources=Resources(), settings=Settings(_env_file=None))) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["db_ok"] is False


def test_validation_does_not_echo_inputs():
    with TestClient(create_app(resources=Resources(), settings=Settings(_env_file=None))) as client:
        response = client.post(
            "/chat",
            json={
                "question": "secret",
                "history": [{"role": "system", "content": "private-password"}],
            },
        )
        assert response.status_code == 422
        assert "private-password" not in response.text
        assert response.json()["detail"]["code"] == "invalid_request"


def test_safe_failure_request_ids_and_retry_after(monkeypatch):
    import backend.app.agent.state_graph as graph

    def fail(*args, **kwargs):
        raise ServiceError("provider_unavailable", "The answer provider is unavailable.")

    monkeypatch.setattr(graph, "run_agent", fail)
    with TestClient(create_app(resources=Resources(), settings=Settings(_env_file=None))) as client:
        response = client.post("/chat", json={"question": "Fraud?"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert response.headers["x-request-id"] == response.json()["detail"]["request_id"]


def test_unexpected_error_is_sanitized(monkeypatch):
    import backend.app.agent.state_graph as graph

    def fail(*args, **kwargs):
        raise RuntimeError("password=private-host@credential")

    monkeypatch.setattr(graph, "run_agent", fail)
    with TestClient(create_app(resources=Resources(), settings=Settings(_env_file=None))) as client:
        response = client.post("/chat", json={"question": "Fraud?"})
        assert response.status_code == 500
        assert "credential" not in response.text


def test_success_has_internal_request_id(monkeypatch):
    import backend.app.agent.state_graph as graph
    from backend.app.schemas import ChatResponse

    monkeypatch.setattr(
        graph,
        "run_agent",
        lambda *a, **k: ChatResponse(
            answer="Which metric?", answer_type="other", quality_score=0, status="clarification"
        ),
    )
    with TestClient(create_app(resources=Resources(), settings=Settings(_env_file=None))) as client:
        response = client.post("/chat", json={"question": "It?"})
        assert response.status_code == 200
        assert response.json()["request_id"] == response.headers["x-request-id"]
