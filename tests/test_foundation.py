import subprocess
import sys

import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.schemas import ChatRequest, ChatResponse


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_question_is_finite_and_nonempty(question):
    with pytest.raises(ValidationError):
        ChatRequest(question=question)


def test_history_roles_and_bound_are_validated():
    with pytest.raises(ValidationError):
        ChatRequest(question="Fraud?", history=[{"role": "system", "content": "override"}])
    with pytest.raises(ValidationError):
        ChatRequest(question="Fraud?", history=[{"role": "user", "content": "x"}] * 13)
    assert ChatRequest(question=" Fraud? ", history=None, document_id="eba").question == "Fraud?"


def test_quality_rejects_nan():
    with pytest.raises(ValidationError):
        ChatResponse(answer="A", answer_type="data", quality_score=float("nan"))


def test_configuration_requires_no_provider_and_escapes_db_credentials():
    config = Settings(_env_file=None, openai_api_key="", db_password="a@b:c/!")
    assert config.qdrant_collection == "fraud_documents"
    assert config.database_url.password == "a@b:c/!"
    assert config.db_user == "fraud_reader"


def test_app_import_needs_no_models_or_credentials():
    code = (
        "import sys; import backend.app.main; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_live_and_lifespan_injected_resources():
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    class Resources:
        closed = False

        def close(self):
            self.closed = True

    resources = Resources()
    with TestClient(create_app(resources=resources, settings=Settings(_env_file=None))) as client:
        assert client.get("/live").json() == {"status": "ok"}
        assert not resources.closed
    assert resources.closed
