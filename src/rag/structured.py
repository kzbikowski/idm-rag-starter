"""Structured output: odpowiedź LLM zgodna z JSON Schema (model Pydantic).

Trzy sposoby, od najmocniejszego do najsłabszego:

1. `native` – `response_format={"type": "json_schema", ...}`: serwer ogranicza dekodowanie
   (constrained decoding / grammar), więc model FIZYCZNIE nie może wygenerować niepoprawnego JSON-a.
2. `tool`   – schemat jako parametry narzędzia `submit_answer` + wymuszony `tool_choice`.
   Modele są trenowane do generowania poprawnych argumentów narzędzi, ale serwer może zignorować
   `tool_choice` – wtedy model odpowie zwykłym tekstem i traktujemy to jako błąd trybu.
3. `prompt` – schemat wklejony do promptu + walidacja Pydantic + 1 ponowienie z opisem błędu.
   Działa wszędzie, ale to tylko „prośba” – stąd walidacja.

`STRUCTURED_OUTPUT=auto` próbuje native, a gdy serwer go nie obsługuje – prompt. Tryb `tool`
jest dostępny tylko jawnie (`STRUCTURED_OUTPUT=tool`): na Ollamie z MLX był wolniejszy od `prompt`
przy tej samej skuteczności. Wykryty tryb zapisujemy na dysku
(`.cache/structured_output.json`, klucz: base_url + model), więc kolejne uruchomienia CLI nie płacą
za ponowne próby. Rozróżniamy dwa rodzaje porażek:
- tryb NIEDOSTĘPNY (serwer zwraca błąd HTTP, np. 501) – zapisujemy w cache i więcej go nie próbujemy;
- pojedyncza NIEUDANA odpowiedź (model nie wywołał narzędzia, zły JSON) – schodzimy niżej tylko
  dla tego pytania, cache się nie zmienia.
Reset cache: `make reset-cache` (np. po aktualizacji Ollamy).
Na Macu z Ollamą (runner MLX) tryb `native` zwraca błąd 501 „structured output is unavailable”.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from openai import APIStatusError, OpenAI
from pydantic import BaseModel, ValidationError

from rag.config import Settings
from rag.llm import chat_kwargs

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, Any]]

TOOL_NAME = "submit_answer"
AUTO_ORDER = ("native", "prompt")

CACHE_PATH = Path(".cache/structured_output.json")
# Kody HTTP oznaczające „serwer nie obsługuje tej funkcji” (a nie chwilową awarię).
UNSUPPORTED_STATUS = {400, 404, 422, 501}


class StructuredOutputError(RuntimeError):
    """Dany tryb nie dał poprawnej odpowiedzi (przechodzimy do następnego)."""


def _native(client: OpenAI, settings: Settings, messages: Messages, schema: type[T]) -> T:
    response = client.chat.completions.create(
        **chat_kwargs(settings),
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    )
    return schema.model_validate_json(response.choices[0].message.content or "")


def _tool(client: OpenAI, settings: Settings, messages: Messages, schema: type[T]) -> T:
    tool = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Zwróć ostateczną odpowiedź w wymaganym formacie.",
            "parameters": schema.model_json_schema(),
        },
    }
    # Część serwerów (np. Ollama) nie wymusza `tool_choice` – dodatkowa instrukcja w prompcie
    # wyraźnie zwiększa szansę, że model faktycznie wywoła narzędzie.
    nudge = {"role": "system", "content": f"Odpowiedz, wywołując narzędzie {TOOL_NAME}."}
    response = client.chat.completions.create(
        **chat_kwargs(settings),
        messages=[*messages, nudge],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
    )
    calls = response.choices[0].message.tool_calls or []
    call = next((c for c in calls if c.function.name == TOOL_NAME), None)
    if call is None:
        raise StructuredOutputError(
            f"model nie wywołał {TOOL_NAME} (serwer zignorował tool_choice)"
        )
    return schema.model_validate_json(call.function.arguments)


def extract_json(text: str) -> str:
    """Wyciąga obiekt JSON z odpowiedzi tekstowej (usuwa ```json ... ``` i tekst wokół)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise StructuredOutputError("w odpowiedzi nie ma obiektu JSON")
    return text[start : end + 1]


def _prompt(client: OpenAI, settings: Settings, messages: Messages, schema: type[T]) -> T:
    instruction = (
        "Odpowiedz WYŁĄCZNIE jednym obiektem JSON zgodnym z tym JSON Schema "
        "(bez komentarzy i tekstu poza JSON-em):\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    attempt: Messages = [*messages, {"role": "system", "content": instruction}]
    for retry in range(2):  # pierwsza próba + 1 ponowienie
        response = client.chat.completions.create(**chat_kwargs(settings), messages=attempt)
        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(extract_json(content))
        except (ValidationError, StructuredOutputError) as err:
            if retry == 1:
                raise StructuredOutputError(f"niepoprawny JSON po ponowieniu: {err}") from err
            attempt = [
                *attempt,
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Ten JSON jest niepoprawny: {err}. Popraw go."},
            ]
    raise AssertionError("unreachable")


_MODES: dict[str, Callable[[OpenAI, Settings, Messages, type[Any]], Any]] = {
    "native": _native,
    "tool": _tool,
    "prompt": _prompt,
}


# --------------------------------------------------------------------------- cache trybu


def _cache_key(settings: Settings) -> str:
    return f"{settings.llm_base_url}|{settings.llm_model}"


def _read_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cached_mode(settings: Settings) -> str | None:
    """Tryb zapisany w cache dla bieżącego base_url + model (None, jeśli brak)."""
    return _read_cache().get(_cache_key(settings), {}).get("mode")


def _save_mode(settings: Settings, mode: str) -> None:
    cache = _read_cache()
    cache[_cache_key(settings)] = {"mode": mode}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_cache() -> bool:
    """Usuwa cache wykrytych trybów. Zwraca True, jeśli plik istniał."""
    existed = CACHE_PATH.exists()
    CACHE_PATH.unlink(missing_ok=True)
    return existed


# --------------------------------------------------------------------------- łańcuch trybów


def complete_structured(
    client: OpenAI,
    settings: Settings,
    messages: Messages,
    schema: type[T],
) -> tuple[T, str]:
    """Zwraca (obiekt zgodny ze `schema`, nazwa trybu, który zadziałał)."""
    if settings.structured_output != "auto":
        order: tuple[str, ...] = (settings.structured_output,)
        start_mode = None
    else:
        start_mode = cached_mode(settings)
        order = (
            AUTO_ORDER[AUTO_ORDER.index(start_mode) :] if start_mode in AUTO_ORDER else AUTO_ORDER
        )

    unavailable: set[str] = set()
    chain: list[str] = []  # do jednej linii logu, np. "native niedostępny → prompt"
    errors: list[str] = []
    for mode in order:
        try:
            result = _MODES[mode](client, settings, messages, schema)
        except APIStatusError as err:
            if err.status_code not in UNSUPPORTED_STATUS:
                raise  # np. 500/503 – awaria serwera, a nie brak funkcji
            unavailable.add(mode)
            chain.append(f"{mode} niedostępny")
            errors.append(f"{mode}: HTTP {err.status_code} {err.message[:150]}")
            log.debug("structured output: %s niedostępny: %s", mode, err.message)
            continue
        except (ValidationError, StructuredOutputError, json.JSONDecodeError) as err:
            chain.append(f"{mode} nieudany")
            errors.append(f"{mode}: {str(err)[:150]}")
            log.debug("structured output: %s – niepoprawna odpowiedź: %s", mode, err)
            continue

        if settings.structured_output == "auto":
            # Najwyższy tryb, który serwer obsługuje (pojedyncze nieudane odpowiedzi go nie obniżają).
            best = next(m for m in order if m not in unavailable)
            if best != start_mode:
                _save_mode(settings, best)
                log.info(
                    "structured output: %s (zapisano tryb %s w %s)",
                    " → ".join([*chain, mode]),
                    best,
                    CACHE_PATH,
                )
            elif chain:
                log.info("structured output: %s", " → ".join([*chain, mode]))
        return result, mode
    raise StructuredOutputError("żaden tryb structured output nie zadziałał:\n" + "\n".join(errors))
