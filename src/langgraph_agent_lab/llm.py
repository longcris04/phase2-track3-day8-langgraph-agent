"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

This lab is configured to use **OpenRouter** (https://openrouter.ai) as the LLM
gateway. OpenRouter exposes an OpenAI-compatible API, so we reuse the
``langchain-openai`` ``ChatOpenAI`` client and simply point it at the OpenRouter
base URL. The model is read from ``OPENROUTER_MODEL`` in ``.env``.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Load .env once at import time so every entrypoint (CLI, pytest, scripts)
# sees the configured keys without having to call load_dotenv() itself.
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM client from environment configuration.

    Resolution order (OpenRouter is preferred for this lab):
    1. OPENROUTER_API_KEY → ChatOpenAI pointed at the OpenRouter gateway
       (model from OPENROUTER_MODEL)
    2. GEMINI_API_KEY     → ChatGoogleGenerativeAI
    3. OPENAI_API_KEY     → ChatOpenAI (api.openai.com)
    4. ANTHROPIC_API_KEY  → ChatAnthropic

    Override the model with the `model` parameter, or the relevant *_MODEL env var.
    """
    # 1. OpenRouter (OpenAI-compatible) — the configured provider for this lab.
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            # Optional OpenRouter ranking headers — harmless if omitted.
            default_headers={
                "HTTP-Referer": "https://github.com/ai-in-action/langgraph-agent-lab",
                "X-Title": "Day08 LangGraph Agent Lab",
            },
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set OPENROUTER_API_KEY (preferred), GEMINI_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
