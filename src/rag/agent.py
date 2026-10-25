"""Prosty agent z tool callingiem. Pętla agenta jest napisana jawnie, bez frameworka.

Schemat pętli:
    1. Wyślij historię rozmowy + definicje narzędzi do LLM.
    2. Jeśli model zwrócił `tool_calls` → wykonaj każde narzędzie, dopisz wynik jako
       wiadomość `role="tool"` i wróć do kroku 1.
    3. Jeśli model zwrócił zwykły tekst → to odpowiedź końcowa.
Limit kroków chroni przed zapętleniem (model może bez końca wołać narzędzia).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rag.config import Settings
from rag.llm import chat_kwargs, llm_client
from rag.retrieve import retrieve

SYSTEM_PROMPT = """Jesteś asystentem, który odpowiada na pytania o dokumenty z bazy wiedzy.
Zanim odpowiesz, użyj narzędzia search_docs (możesz kilka razy, z różnymi zapytaniami).
Odpowiadaj po polsku, tylko na podstawie wyników wyszukiwania, i podawaj źródła w nawiasach,
np. (rag_wprowadzenie.md › Czym jest RAG). Jeśli wyniki nie zawierają odpowiedzi, powiedz to wprost."""

SEARCH_DOCS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Przeszukuje bazę dokumentów i zwraca najbardziej pasujące fragmenty.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Zapytanie wyszukiwania."},
                "k": {"type": "integer", "description": "Liczba fragmentów (1-10).", "default": 4},
            },
            "required": ["query"],
        },
    },
}


def search_docs(settings: Settings, query: str, k: int = 4) -> str:
    """Implementacja narzędzia: retrieval i sformatowanie wyników jako tekst dla modelu."""
    hits = retrieve(query, settings, k=max(1, min(int(k), 10)))
    if not hits:
        return "Brak wyników."
    return "\n\n".join(f"Źródło: {h.label()}\n{h.text}" for h in hits)


@dataclass
class AgentStep:
    """Jeden krok śladu agenta (do wyświetlenia w terminalu)."""

    kind: str  # "tool_call" | "answer"
    content: str
    arguments: dict[str, Any] = field(default_factory=dict)


def run_agent(
    question: str,
    settings: Settings,
    max_steps: int = 5,
    on_step: Callable[[AgentStep], None] | None = None,
) -> str:
    """Uruchamia pętlę agenta i zwraca odpowiedź końcową; każdy krok przekazuje do `on_step`."""
    client = llm_client(settings)
    tools: dict[str, Callable[..., str]] = {"search_docs": lambda **a: search_docs(settings, **a)}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    def record(step: AgentStep) -> None:
        if on_step:
            on_step(step)

    for _ in range(max_steps):
        response = client.chat.completions.create(
            **chat_kwargs(settings), messages=messages, tools=[SEARCH_DOCS_TOOL]
        )
        message = response.choices[0].message

        if not message.tool_calls:  # brak wywołań narzędzi → odpowiedź końcowa
            answer = message.content or ""
            record(AgentStep("answer", answer))
            return answer

        # Odpowiedź asystenta z tool_calls musi trafić do historii PRZED wynikami narzędzi.
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            tool = tools.get(call.function.name)
            result = tool(**args) if tool else f"Nieznane narzędzie: {call.function.name}"
            record(AgentStep("tool_call", result, {"name": call.function.name, **args}))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    answer = "Przekroczono limit kroków agenta – nie udało się ustalić odpowiedzi."
    record(AgentStep("answer", answer))
    return answer
