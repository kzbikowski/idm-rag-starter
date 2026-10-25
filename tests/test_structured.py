"""Łańcuch trybów structured output i cache na dysku – na atrapie klienta OpenAI (bez sieci)."""

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, InternalServerError
from pydantic import BaseModel

from rag import structured
from rag.config import load_settings
from rag.structured import StructuredOutputError, cached_mode, complete_structured, extract_json


class Out(BaseModel):
    answer: str


def _message(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


GOOD_TOOL = _message(tool_calls=[_tool_call("submit_answer", '{"answer": "ok"}')])
IGNORED_TOOL = _message(content="Odpowiedź tekstem, bez narzędzia")


class FakeClient:
    """Atrapa: native → 501 (albo `native_reply`), tool → `tool_reply`, prompt → `prompt_replies`."""

    def __init__(self, tool_reply, prompt_replies=('{"answer": "42"}',) * 5):
        self.tool_reply = tool_reply
        self.prompt_replies = list(prompt_replies)
        self.native_reply = None  # None = serwer nie obsługuje native (HTTP 501)
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if "response_format" in kwargs:
            self.calls.append("native")
            if self.native_reply is not None:
                return self.native_reply
            request = httpx.Request("POST", "http://x")
            raise InternalServerError(
                "structured output is unavailable",
                response=httpx.Response(501, request=request),
                body=None,
            )
        if "tools" in kwargs:
            self.calls.append("tool")
            return self.tool_reply
        self.calls.append("prompt")
        return _message(content=self.prompt_replies.pop(0))


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("STRUCTURED_OUTPUT", "auto")
    monkeypatch.setattr(structured, "CACHE_PATH", tmp_path / "structured_output.json")
    return load_settings(env_file=None)


def test_native_unavailable_falls_back_to_prompt_and_caches_it(settings):
    client = FakeClient(GOOD_TOOL)
    result, mode = complete_structured(client, settings, [], Out)
    assert (result.answer, mode) == ("42", "prompt")
    assert client.calls == ["native", "prompt"]  # auto nie próbuje trybu tool
    assert cached_mode(settings) == "prompt"


def test_cache_skips_unavailable_native_in_next_process(settings):
    complete_structured(FakeClient(GOOD_TOOL), settings, [], Out)
    client = FakeClient(GOOD_TOOL)  # „nowy proces”: nowa atrapa, ten sam plik cache
    complete_structured(client, settings, [], Out)
    assert client.calls == ["prompt"]


def test_invalid_native_answer_falls_back_but_keeps_native_in_cache(settings):
    client = FakeClient(GOOD_TOOL)
    client.native_reply = _message(content="to nie JSON")
    _, mode = complete_structured(client, settings, [], Out)
    assert mode == "prompt"
    assert cached_mode(settings) == "native"  # serwer obsługuje native, zawiodła jedna odpowiedź


def test_explicit_tool_mode(settings):
    tool_only = replace(settings, structured_output="tool")
    result, mode = complete_structured(FakeClient(GOOD_TOOL), tool_only, [], Out)
    assert (result.answer, mode) == ("ok", "tool")
    with pytest.raises(StructuredOutputError, match="submit_answer"):
        complete_structured(FakeClient(IGNORED_TOOL), tool_only, [], Out)


def test_connection_error_is_not_treated_as_unsupported(settings):
    client = FakeClient(GOOD_TOOL)

    def down(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    client.chat.completions.create = down
    with pytest.raises(APIConnectionError):
        complete_structured(client, settings, [], Out)
    assert cached_mode(settings) is None


def test_prompt_mode_retries_once_on_invalid_json(settings):
    replies = ("nie json", '```json\n{"answer": "a"}\n```')
    client = FakeClient(IGNORED_TOOL, prompt_replies=replies)
    prompt_only = replace(settings, structured_output="prompt")
    result, mode = complete_structured(client, prompt_only, [], Out)
    assert (result.answer, mode) == ("a", "prompt")
    assert client.calls == ["prompt", "prompt"]
    assert cached_mode(settings) is None  # tryb wymuszony w .env nie trafia do cache


def test_reset_cache(settings):
    complete_structured(FakeClient(GOOD_TOOL), settings, [], Out)
    assert structured.reset_cache() is True
    assert cached_mode(settings) is None
    assert structured.reset_cache() is False


def test_extract_json():
    assert extract_json('Oto wynik: {"a": 1} dzięki') == '{"a": 1}'
    with pytest.raises(StructuredOutputError):
        extract_json("brak")
