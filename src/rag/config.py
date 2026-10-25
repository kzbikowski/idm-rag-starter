"""Konfiguracja projektu wczytywana z pliku `.env` i zmiennych środowiskowych.

Cała konfiguracja jest w jednym miejscu, żeby było jasne, co można zmienić bez dotykania kodu.
Przełączenie z lokalnej Ollamy na Gemini API to wyłącznie zmiana `.env`
(obie usługi mają API zgodne z OpenAI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv

STRUCTURED_OUTPUT_MODES = ("auto", "native", "tool", "prompt")
CHUNKING_STRATEGIES = ("fixed", "recursive")


def _env(name: str, default: str) -> str:
    """Zwraca zmienną środowiskową albo wartość domyślną (pusty string traktujemy jak brak)."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Wszystkie ustawienia projektu. Nazwy pól odpowiadają zmiennym w `.env`."""

    # --- LLM (chat completions) ---
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # "none" wyłącza tryb thinking w qwen3.5 (w Ollamie: ~100x krótszy czas odpowiedzi).
    # Pusty string = nie wysyłaj parametru (np. gdy provider go nie obsługuje).
    llm_reasoning_effort: str
    llm_temperature: float

    # --- Embedding ---
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    # Instrukcja doklejana do zapytań (nie do dokumentów) – qwen3-embedding jest trenowany
    # asymetrycznie. Pusty string = brak instrukcji (np. dla modeli Gemini).
    embed_query_instruction: str

    # --- Dane ---
    data_dir: str

    # --- Vector store ---
    qdrant_url: str
    collection: str

    # --- RAG ---
    top_k: int
    chunking: str
    chunk_size: int
    chunk_overlap: int
    structured_output: str

    def validate(self) -> None:
        """Sprawdza wartości, które łatwo wpisać błędnie, i rzuca czytelny błąd."""
        if self.structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"STRUCTURED_OUTPUT={self.structured_output!r}; dozwolone: {STRUCTURED_OUTPUT_MODES}"
            )
        if self.chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"CHUNKING={self.chunking!r}; dozwolone: {CHUNKING_STRATEGIES}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP musi być mniejszy niż CHUNK_SIZE")

    @property
    def is_local_ollama(self) -> bool:
        """Czy LLM działa lokalnie (wtedy `make check` sprawdza też Ollamę i pobrane modele)."""
        host = urlparse(self.llm_base_url).hostname
        return host in ("localhost", "127.0.0.1")


def load_settings(env_file: str | None = ".env") -> Settings:
    """Wczytuje `.env` (jeśli istnieje) i buduje obiekt `Settings`.

    Zmienne ustawione w powłoce mają pierwszeństwo przed `.env`
    (np. `LLM_MODEL=qwen3.5:9b make ask Q="..."`).
    """
    if env_file:
        load_dotenv(env_file, override=False)

    llm_base_url = _env("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key = _env("LLM_API_KEY", "ollama")
    settings = Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=_env("LLM_MODEL", "qwen3.5:4b"),
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "none"),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0")),
        embed_base_url=_env("EMBED_BASE_URL", llm_base_url),
        embed_api_key=_env("EMBED_API_KEY", llm_api_key),
        embed_model=_env("EMBED_MODEL", "qwen3-embedding:0.6b"),
        embed_query_instruction=os.getenv(
            "EMBED_QUERY_INSTRUCTION",
            "Given a question, retrieve passages from the documents that answer it",
        ),
        data_dir=_env("DATA_DIR", "data"),
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        collection=_env("COLLECTION", "docs"),
        top_k=int(_env("TOP_K", "5")),
        chunking=_env("CHUNKING", "recursive"),
        chunk_size=int(_env("CHUNK_SIZE", "800")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "120")),
        structured_output=_env("STRUCTURED_OUTPUT", "auto"),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Ustawienia wczytane raz na proces (wygodne w CLI)."""
    return load_settings()
