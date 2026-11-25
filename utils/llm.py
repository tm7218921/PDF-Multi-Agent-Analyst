from __future__ import annotations

from typing import List, Dict

from openai import OpenAI

from .groq_config import GROQ_BASE_URL, get_groq_api_key, get_groq_model

_client: OpenAI | None = None

MODEL = get_groq_model()
MAX_TOKENS = 2048
TEMPERATURE = 0.2


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_groq_api_key(),
            base_url=GROQ_BASE_URL,
        )
    return _client


def call_llm(
    messages: List[Dict[str, str]],
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Send a list of chat messages to the Groq API and return the reply text.

    messages format: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc
