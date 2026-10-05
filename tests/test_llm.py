from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.config import Settings


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def test_llm_honors_task_model_limits_and_json_schema():
    from backend.app.llm.openai_client import LLMClient
    from backend.app.runtime import Deadline

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"value":"ok"}', refusal=None),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = LLMClient(Settings(_env_file=None, openai_model_name="chosen-answer"), client=client)
    assert llm.complete("answer", Output, "system", "user", Deadline(3)).value == "ok"
    assert calls[0]["model"] == "chosen-answer"
    assert 0 < calls[0]["timeout"] <= 3
    assert calls[0]["max_completion_tokens"] == 4096
    assert calls[0]["response_format"]["json_schema"]["strict"]
    assert "temperature" not in calls[0]
    assert "reasoning_effort" not in calls[0]


def test_bad_or_incomplete_provider_output_is_safe():
    from backend.app.llm.openai_client import LLMClient
    from backend.app.runtime import Deadline, ServiceError

    def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="secret output", refusal=None),
                )
            ]
        )

    llm = LLMClient(
        Settings(_env_file=None),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    with pytest.raises(ServiceError) as exc:
        llm.complete("answer", Output, "S", "U", Deadline(3))
    assert "secret output" not in str(exc.value)


@pytest.mark.parametrize(
    "task,model,effort,tokens",
    [
        ("answer", "gpt-5-mini", "low", 8192),
        ("sql", "gpt-5-mini", "low", 8192),
        ("planner", "gpt-5-nano", "minimal", 2048),
        ("judge", "gpt-5-nano", "minimal", 2048),
    ],
)
def test_pinned_sdk_wire_contract_without_network(task, model, effort, tokens):
    import json

    import httpx2
    from openai import OpenAI

    from backend.app.llm.openai_client import LLMClient
    from backend.app.runtime import Deadline

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"value":"verified"}'},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="offline-fixture",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    ) as sdk:
        result = LLMClient(Settings(_env_file=None), client=sdk).complete(
            task, Output, "System", "User", Deadline(3)
        )
    assert result.value == "verified"
    assert bodies[0]["model"] == model
    assert bodies[0]["reasoning_effort"] == effort
    assert bodies[0]["max_completion_tokens"] == tokens
    assert bodies[0]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
