# backend/app/llm/client.py

from typing import Optional

from openai import OpenAI

from ..config import get_settings

settings = get_settings()

_client = OpenAI(api_key=settings.openai_api_key)


def call_gpt5_mini(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Generic wrapper to call the GPT-5 Mini model (or any model name provided).

    Uses openai==2.9.0 style:
    client = OpenAI()
    client.chat.completions.create(...)
    """
    completion = _client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=temperature,
        # max_completion_tokens=max_tokens,
    )

    content: Optional[str] = completion.choices[0].message.content
    # print("GPT-5 Mini is Called")
    return (content or "").strip()


def call_gpt5_nano(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Generic wrapper to call the GPT-5 Nano model (or any model name provided).

    Uses openai==2.9.0 style:
    client = OpenAI()
    client.chat.completions.create(...)
    """
    completion = _client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=temperature,
        # max_completion_tokens=max_tokens,
    )

    content: Optional[str] = completion.choices[0].message.content
    # print("GPT-5 Nano is Called")
    return (content or "").strip()