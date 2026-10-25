"""Embeddingi przez endpoint `/v1/embeddings` zgodny z OpenAI.

Embedding zapytania i dokumentu liczymy INACZEJ: modele takie jak qwen3-embedding są trenowane
asymetrycznie – zapytanie dostaje instrukcję opisującą zadanie, dokument nie. Pominięcie
instrukcji zwykle obniża jakość retrievalu o kilka punktów procentowych.
"""

from __future__ import annotations

from rag.config import Settings
from rag.llm import embed_client

BATCH_SIZE = 32


def embed_documents(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embeddingi dokumentów (chunków), wysyłane w paczkach po `BATCH_SIZE`."""
    client = embed_client(settings)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=settings.embed_model, input=batch)
        vectors += [item.embedding for item in response.data]
    return vectors


def format_query(query: str, instruction: str) -> str:
    """Format zapytania zalecany dla qwen3-embedding: `Instruct: ...\\nQuery: ...`."""
    if not instruction:
        return query
    return f"Instruct: {instruction}\nQuery: {query}"


def embed_query(query: str, settings: Settings) -> list[float]:
    """Embedding pojedynczego zapytania (z instrukcją, jeśli skonfigurowana)."""
    client = embed_client(settings)
    text = format_query(query, settings.embed_query_instruction)
    response = client.embeddings.create(model=settings.embed_model, input=[text])
    return response.data[0].embedding
