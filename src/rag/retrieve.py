"""Retrieval: zapytanie → embedding → top-k chunków z Qdranta.

Tu są też miejsca do rozbudowy (TODO) – kolejne techniki poprawy retrievalu omawiane na kursie.
Każde TODO jest opisane: co zrobić, gdzie, jak sprawdzić efekt (`make eval`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from rag.config import Settings
from rag.embeddings import embed_query
from rag.store import get_client, search


@dataclass
class Hit:
    """Wynik retrievalu: chunk z metadanymi i podobieństwem (cosine similarity)."""

    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def source_ids(self) -> list[str]:
        return list(self.metadata.get("source_ids", []))

    def label(self) -> str:
        """Czytelna etykieta źródła, np. `regulamin.pdf, s. 3` albo `notatki.md › Chunking`."""
        m = self.metadata
        label = str(m.get("source", "?"))
        if m.get("page"):
            label += f", s. {m['page']}"
        if m.get("section"):
            label += f" › {m['section']}"
        return label


def retrieve(
    query: str,
    settings: Settings,
    k: int | None = None,
    collection: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[Hit]:
    """Zwraca top-k chunków najbardziej podobnych do zapytania."""
    k = k or settings.top_k
    collection = collection or settings.collection
    client = get_client(settings.qdrant_url)

    # TODO(query-rewriting): query = rewrite_query(query, settings)
    vector = embed_query(query, settings)
    points = search(client, collection, vector, k=k, query_filter=build_filter(filters))
    # TODO(hybrid): zamiast `search` wywołaj `hybrid_search(...)`.
    # TODO(reranking): pobierz więcej kandydatów (np. 20) i wywołaj `rerank(query, hits, k)`.

    hits = []
    for p in points:
        payload = dict(p.payload or {})
        hits.append(Hit(text=payload.pop("text", ""), score=p.score, metadata=payload))
    _check_embed_model(hits, settings)
    return hits


def _check_embed_model(hits: list[Hit], settings: Settings) -> None:
    """Wektory w kolekcji muszą pochodzić z TEGO SAMEGO modelu co embedding zapytania."""
    indexed_with = hits[0].metadata.get("embed_model") if hits else None
    if indexed_with and indexed_with != settings.embed_model:
        raise RuntimeError(
            f"Kolekcja zindeksowana modelem {indexed_with!r}, a EMBED_MODEL={settings.embed_model!r}. "
            "Embeddingi z różnych modeli są nieporównywalne – uruchom: make ingest ARGS=--recreate"
        )


# =============================================================================== TODO
# Poniższe funkcje są szkieletami do samodzielnej implementacji. Po każdej zmianie
# uruchom `make eval` i porównaj recall@k / MRR z wersją bazową (eval/results/).


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Zamienia słownik `{pole: wartość}` na filtr Qdranta (metadata filtering).

    Zaimplementowany jest tylko najprostszy przypadek: równość dla pól tekstowych.

    TODO(metadata-filtering):
    - listy wartości → `models.MatchAny(any=[...])`;
    - zakresy (np. strony, daty) → `models.FieldCondition(key=..., range=models.Range(gte=...))`;
    - flaga `--filter pole=wartość` w `cli.py` (komenda `ask`) przekazywana do `retrieve()`;
    - indeks payloadu dla nowych pól w `store.ensure_collection` (szybsze filtrowanie).
    Jak sprawdzić: `make ask Q="..." ARGS="--filter source=rag_wprowadzenie.md"` – wszystkie
    źródła w odpowiedzi muszą pochodzić z wybranego pliku.
    """
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def hybrid_search(query: str, settings: Settings, k: int) -> list[Hit]:
    """TODO(hybrid): hybrid search = dense (embeddingi) + sparse (BM25), fuzja RRF.

    Dlaczego: embeddingi gubią dokładne dopasowania (numery paragrafów, nazwy własne, skróty),
    a BM25 (bag of words) radzi sobie z nimi świetnie. Połączenie zwykle poprawia recall.

    Co zrobić:
    1. `uv add fastembed` (dopisuje zależność do pyproject.toml).
    2. W `store.ensure_collection` dodaj `sparse_vectors_config={"bm25":
       models.SparseVectorParams(modifier=models.Modifier.IDF)}`.
    3. W ingeście licz wektory sparse: `from fastembed import SparseTextEmbedding;
       SparseTextEmbedding("Qdrant/bm25")` i zapisuj je obok dense:
       `vector={"dense": [...], "bm25": models.SparseVector(indices=..., values=...)}`.
       Uwaga: model "Qdrant/bm25" domyślnie używa stemmera angielskiego – sprawdź parametr
       `language` i porównaj wyniki dla polskiego tekstu.
    4. Tutaj użyj Query API z fuzją:
       `client.query_points(collection, prefetch=[models.Prefetch(query=dense, using="dense",
       limit=20), models.Prefetch(query=sparse, using="bm25", limit=20)],
       query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k)`.
    Jak sprawdzić: `make eval` przed i po – szczególnie pytania z numerami/nazwami.
    """
    raise NotImplementedError("TODO(hybrid): zobacz docstring hybrid_search()")


def rerank(query: str, hits: list[Hit], k: int) -> list[Hit]:
    """TODO(reranking): przesortuj kandydatów cross-encoderem `BAAI/bge-reranker-v2-m3`.

    Dlaczego: bi-encoder (embedding) koduje pytanie i chunk osobno; cross-encoder czyta je
    RAZEM, więc ocenia trafność dokładniej – ale jest wolny, dlatego stosujemy go tylko do
    top-N (np. 20) kandydatów z retrievalu.

    Co zrobić:
    1. Wybierz backend: `fastembed.rerank.cross_encoder.TextCrossEncoder` (jeśli wspiera
       ten model w Twojej wersji) albo `sentence-transformers` `CrossEncoder(...)` (cięższy, torch).
    2. Policz `scores = model.rerank(query, [h.text for h in hits])`.
    3. Zwróć `k` hitów o najwyższym score (nadpisz `Hit.score`).
    4. W `retrieve()` pobieraj `k * 4` kandydatów i wywołuj `rerank`.
    Jak sprawdzić: MRR w `make eval` powinien wzrosnąć; zmierz też narzut czasowy.
    """
    raise NotImplementedError("TODO(reranking): zobacz docstring rerank()")


def rewrite_query(query: str, settings: Settings) -> str:
    """TODO(query-rewriting): przepisz pytanie użytkownika na lepsze zapytanie do retrievalu.

    Dlaczego: pytania bywają potoczne, wieloczęściowe lub zależne od kontekstu rozmowy
    („a jak to wygląda na studiach II stopnia?”). Embedding takiego pytania słabo trafia.

    Co zrobić (wybierz wariant):
    - rewrite: poproś LLM (`llm.llm_client`, temperature=0) o jedno zwięzłe zapytanie
      w języku dokumentów, bez grzeczności i dygresji;
    - multi-query: wygeneruj 3 warianty, zrób retrieval dla każdego i połącz wyniki (RRF);
    - HyDE: wygeneruj hipotetyczną odpowiedź i zrób embedding JEJ zamiast pytania.
    Jak sprawdzić: `make eval`; zwróć uwagę na pytania wymagające połączenia kilku źródeł.
    """
    raise NotImplementedError("TODO(query-rewriting): zobacz docstring rewrite_query()")
