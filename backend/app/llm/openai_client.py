"""Structured LLM boundary with explicit models, budgets and bounded retries."""

import re
import time
from threading import RLock

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from ..runtime import ServiceError


class LLMClient:
    def __init__(self, settings, *, client=None):
        self.settings = settings
        self._client = client
        self._lock = RLock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                key = self.settings.openai_api_key.get_secret_value()
                if not key:
                    raise ServiceError(
                        "provider_unconfigured",
                        "Configure the answer provider first.",
                        retryable=False,
                    )
                self._client = OpenAI(
                    api_key=key, max_retries=0, timeout=self.settings.external_timeout_seconds
                )
            return self._client

    def complete(self, task, schema, system, user, deadline):
        model = {
            "answer": self.settings.openai_model_name,
            "sql": self.settings.openai_sql_model,
            "planner": self.settings.openai_planner_model,
            "judge": self.settings.openai_judge_model,
        }[task]
        tokens = {"answer": 4096, "sql": 2048, "planner": 2048, "judge": 2048}[task]
        options = {}
        # Original GPT-5 budgets include hidden reasoning. Reserve more output
        # for SQL/answers and bound effort; omit the parameter for other models.
        if re.fullmatch(r"gpt-5(?:-mini|-nano)?(?:-\d{4}-\d{2}-\d{2})?", model):
            options["reasoning_effort"] = "low" if task in ("sql", "answer") else "minimal"
            if task in ("sql", "answer"):
                tokens = 8192
        client = self._get_client()
        for attempt in range(2):
            deadline.check()
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "strict": True,
                            "schema": schema.model_json_schema(),
                        },
                    },
                    max_completion_tokens=tokens,
                    timeout=deadline.timeout(self.settings.external_timeout_seconds),
                    **options,
                )
                deadline.check()
                choice = completion.choices[0]
                if choice.finish_reason != "stop" or choice.message.refusal:
                    raise ServiceError(
                        "provider_output", "The provider did not complete an answer."
                    )
                return schema.model_validate_json(choice.message.content or "")
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                transient = not isinstance(exc, APIStatusError) or exc.status_code in (
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                )
                if attempt == 0 and transient and deadline.remaining() > 1:
                    time.sleep(min(0.25, deadline.remaining()))
                    continue
                deadline.check()
                raise ServiceError(
                    "provider_unavailable",
                    "The answer provider is unavailable.",
                    retryable=transient,
                ) from None
            except (ValidationError, IndexError, AttributeError):
                raise ServiceError(
                    "provider_output", "The provider returned an invalid answer."
                ) from None
        raise AssertionError("unreachable")

    def close(self):
        if self._client is not None:
            self._client.close()
