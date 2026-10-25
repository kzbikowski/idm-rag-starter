"""Klienci OpenAI SDK dla LLM i embeddingów.

Używamy oficjalnego SDK `openai` z własnym `base_url` – to samo API obsługuje Ollama,
i Gemini (endpoint zgodny z OpenAI). Nie potrzebujemy LangChaina.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from rag.config import Settings


def llm_client(settings: Settings) -> OpenAI:
    """Klient do chat completions.

    `max_retries=0`: przy lokalnej Ollamie ponawianie tylko wydłuża czekanie na błąd,
    a błędy chcemy zobaczyć od razu (np. 501 przy braku structured output).
    """
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=0)


def embed_client(settings: Settings) -> OpenAI:
    """Klient do embeddingów (może wskazywać na innego providera niż LLM)."""
    return OpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key, max_retries=1)


def chat_kwargs(settings: Settings) -> dict[str, Any]:
    """Wspólne parametry każdego wywołania chat completions.

    `reasoning_effort="none"` wyłącza thinking w qwen3.5 (Ollama mapuje go na `think=false`).
    Bez tego model generuje setki tokenów „rozumowania” przed odpowiedzią.
    """
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
    }
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return kwargs
