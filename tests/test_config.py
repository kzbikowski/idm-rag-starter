import pytest

from rag.config import load_settings


def test_defaults(monkeypatch):
    for var in ("LLM_MODEL", "EMBED_BASE_URL", "LLM_BASE_URL", "STRUCTURED_OUTPUT"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(env_file=None)
    assert s.llm_model == "qwen3.5:4b"
    assert s.embed_base_url == s.llm_base_url  # embeddingi domyślnie z tego samego serwera
    assert s.is_local_ollama


def test_invalid_structured_mode(monkeypatch):
    monkeypatch.setenv("STRUCTURED_OUTPUT", "magic")
    with pytest.raises(ValueError, match="STRUCTURED_OUTPUT"):
        load_settings(env_file=None)
