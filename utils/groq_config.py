from __future__ import annotations

import os

from agents import OpenAIProvider, RunConfig
from openai import AsyncOpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
_API_KEY_PLACEHOLDER = "your_groq_api_key_here"


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == _API_KEY_PLACEHOLDER:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add your Groq API key to the .env file."
        )
    return api_key


def groq_api_key_is_configured() -> bool:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(api_key and api_key != _API_KEY_PLACEHOLDER)


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def create_groq_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=get_groq_api_key(),
        base_url=GROQ_BASE_URL,
    )


def create_groq_run_config(client: AsyncOpenAI) -> RunConfig:
    return RunConfig(
        model_provider=OpenAIProvider(
            openai_client=client,
            use_responses=False,
        )
    )
