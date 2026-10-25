"""Smoke test środowiska (`make check`).

Sprawdza po kolei wszystko, co jest potrzebne do działania RAG, i przy błędzie podpowiada,
co zrobić. Kolejne kroki zależą od poprzednich, więc zatrzymujemy się na pierwszym błędzie.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from rich.console import Console

from rag.agent import SEARCH_DOCS_TOOL
from rag.chunking import Chunk
from rag.config import Settings
from rag.embeddings import embed_documents, embed_query
from rag.generate import answer_question
from rag.llm import chat_kwargs, llm_client
from rag.store import ensure_collection, get_client, upsert_chunks
from rag.structured import reset_cache

CHECK_COLLECTION = "__smoke_test__"
MIN_CONTEXT = 8192

console = Console()


class CheckFailed(Exception):
    """Krok nie przeszedł; `hint` mówi, jak to naprawić."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


# Krok = (nazwa, funkcja); funkcja zwraca krótki opis sukcesu albo rzuca CheckFailed.
Step = tuple[str, Callable[[], str]]


def _ollama_root(settings: Settings) -> str:
    return settings.llm_base_url.removesuffix("/").removesuffix("/v1")


def check_ollama(settings: Settings) -> str:
    try:
        version = httpx.get(f"{_ollama_root(settings)}/api/version", timeout=3).json()["version"]
    except httpx.HTTPError as err:
        raise CheckFailed(
            f"Ollama nie odpowiada pod {_ollama_root(settings)} ({err.__class__.__name__})",
            "Uruchom Ollamę: `ollama serve` (lub aplikację Ollama). Mac/brew: `brew services start ollama`.",
        ) from err
    return f"Ollama {version}"


def check_models(settings: Settings) -> str:
    tags = httpx.get(f"{_ollama_root(settings)}/api/tags", timeout=5).json()
    available = {m["name"] for m in tags.get("models", [])}
    available |= {name.removesuffix(":latest") for name in available}
    missing = [m for m in (settings.llm_model, settings.embed_model) if m not in available]
    if missing:
        raise CheckFailed(
            f"Brak modeli: {', '.join(missing)}",
            "Pobierz je: "
            + " && ".join(f"ollama pull {m}" for m in missing)
            + " (albo `make pull`).",
        )
    return f"{settings.llm_model}, {settings.embed_model}"


def check_qdrant(settings: Settings) -> str:
    try:
        client = get_client(settings.qdrant_url)
        n = len(client.get_collections().collections)
    except Exception as err:
        raise CheckFailed(
            f"Qdrant niedostępny pod {settings.qdrant_url}: {err.__class__.__name__}",
            "Uruchom `make up` (Docker/OrbStack musi działać) albo ustaw w .env "
            "QDRANT_URL=local:./.qdrant (tryb bez Dockera).",
        ) from err
    return f"{settings.qdrant_url} ({n} kolekcji)"


def check_embedding(settings: Settings) -> str:
    try:
        vector = embed_query("test", settings)
    except Exception as err:
        raise CheckFailed(
            f"Embedding nie działa: {err}",
            f"Sprawdź EMBED_MODEL={settings.embed_model} i EMBED_BASE_URL={settings.embed_base_url}.",
        ) from err
    return f"wymiar {len(vector)}"


def check_rag(settings: Settings) -> str:
    """Mini-kolekcja z 3 chunkami → jedno pytanie RAG → sprzątanie."""
    texts = [
        "Kot Filemon mieszka w Warszawie i lubi mleko.",
        "Qdrant to baza wektorowa napisana w języku Rust.",
        "Pociąg do Krakowa odjeżdża o 7:15 z Dworca Centralnego.",
    ]
    chunks = [
        Chunk(
            t,
            {
                "source": "smoke.txt",
                "chunk_index": i,
                "chunking": "check",
                "source_ids": ["smoke.txt"],
            },
        )
        for i, t in enumerate(texts)
    ]
    client = get_client(settings.qdrant_url)
    vectors = embed_documents(texts, settings)
    ensure_collection(client, CHECK_COLLECTION, dim=len(vectors[0]), recreate=True)
    upsert_chunks(client, CHECK_COLLECTION, chunks, vectors, settings.embed_model)
    try:
        result = answer_question(
            "W jakim języku napisano Qdranta?", settings, k=2, collection=CHECK_COLLECTION
        )
    except Exception as err:
        raise CheckFailed(
            f"Zapytanie RAG nie powiodło się: {err}",
            f"Sprawdź LLM_MODEL={settings.llm_model} i LLM_BASE_URL={settings.llm_base_url}.",
        ) from err
    finally:
        client.delete_collection(CHECK_COLLECTION)
    if "Qdrant" not in result.hits[0].text:
        raise CheckFailed(
            "Retrieval zwrócił zły chunk na pierwszym miejscu", "Sprawdź EMBED_MODEL."
        )
    elapsed = result.retrieve_s + result.generate_s
    return f"„{result.answer.answer[:60]}” ({elapsed:.1f}s, structured output: {result.mode})"


def check_context_window(settings: Settings) -> str:
    """Ollama po cichu obcina prompt dłuższy niż okno kontekstu – sprawdzamy jego rozmiar."""
    models = httpx.get(f"{_ollama_root(settings)}/api/ps", timeout=5).json().get("models", [])
    loaded = next((m for m in models if m["name"].startswith(settings.llm_model)), None)
    ctx = loaded.get("context_length") if loaded else None
    if ctx is None:
        return "nie udało się odczytać (pomijam)"
    if ctx < MIN_CONTEXT:
        raise CheckFailed(
            f"Okno kontekstu {settings.llm_model} = {ctx} tokenów",
            f"Ustaw OLLAMA_CONTEXT_LENGTH={MIN_CONTEXT * 2} i zrestartuj Ollamę (szczegóły w README).",
        )
    return f"{ctx} tokenów"


def check_tool_call(settings: Settings) -> str:
    response = llm_client(settings).chat.completions.create(
        **chat_kwargs(settings),
        messages=[{"role": "user", "content": "Wyszukaj w dokumentach informacje o urlopie."}],
        tools=[SEARCH_DOCS_TOOL],
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        raise CheckFailed(
            "Model nie wywołał narzędzia search_docs",
            "Użyj modelu z obsługą tool callingu (np. qwen3.5) – sprawdź `ollama show <model>`.",
        )
    return f"{calls[0].function.name}({calls[0].function.arguments})"


def run_checks(settings: Settings) -> bool:
    """Uruchamia wszystkie kroki i wypisuje wynik. Zwraca True, jeśli wszystko przeszło.

    Cache trybu structured output czyścimy, żeby krok RAG wykrył tryb od nowa.
    """
    reset_cache()
    steps: list[Step] = []
    if settings.is_local_ollama:
        steps += [
            ("Ollama działa", lambda: check_ollama(settings)),
            ("Modele pobrane", lambda: check_models(settings)),
        ]
    steps += [
        ("Qdrant działa", lambda: check_qdrant(settings)),
        ("Embedding", lambda: check_embedding(settings)),
        ("Zapytanie RAG", lambda: check_rag(settings)),
        ("Tool calling", lambda: check_tool_call(settings)),
    ]
    if settings.is_local_ollama:
        steps.append(("Okno kontekstu", lambda: check_context_window(settings)))

    console.print(f"[bold]LLM:[/] {settings.llm_model} @ {settings.llm_base_url}")
    for name, run in steps:
        try:
            detail = run()
        except CheckFailed as err:
            console.print(f"❌ {name}: {err}")
            console.print(f"   [yellow]→ {err.hint}[/]")
            return False
        except Exception as err:  # nieoczekiwany błąd – też pokazujemy czytelnie
            console.print(f"❌ {name}: {err.__class__.__name__}: {err}")
            return False
        console.print(f"✅ {name}: {detail}")
    console.print("[bold green]Wszystko działa.[/]")
    return True
