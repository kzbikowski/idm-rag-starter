"""Vector store: Qdrant (serwer z docker-compose albo tryb lokalny bez Dockera).

`QDRANT_URL` może mieć postać:
- `http://localhost:6333` – serwer Qdrant (docker compose up),
- `local:./.qdrant`       – tryb embedded: dane w katalogu, bez Dockera (wolniejszy, jeden proces),
- `:memory:`              – w pamięci (testy).
"""

from __future__ import annotations

import atexit

from qdrant_client import QdrantClient, models

from rag.chunking import Chunk

DENSE = "dense"  # nazwa wektora; TODO(hybrid): obok dodać wektor sparse "bm25"


_clients: dict[str, QdrantClient] = {}


def get_client(url: str) -> QdrantClient:
    """Klient Qdranta dla `QDRANT_URL` (jeden na proces – tryb lokalny blokuje katalog z danymi)."""
    if url not in _clients:
        if url == ":memory:":
            _clients[url] = QdrantClient(location=":memory:")
        elif url.startswith("local:"):
            _clients[url] = QdrantClient(path=url.removeprefix("local:"))
        else:
            _clients[url] = QdrantClient(url=url, timeout=10)
    return _clients[url]


@atexit.register
def close_clients() -> None:
    """Zamyka klientów przy wyjściu (tryb lokalny zapisuje wtedy dane i zwalnia blokadę)."""
    for client in _clients.values():
        client.close()
    _clients.clear()


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    """Tworzy kolekcję (odległość cosinusowa). Przy niezgodnym wymiarze rzuca czytelny błąd."""
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors
        size = existing[DENSE].size if isinstance(existing, dict) else None
        if size != dim:
            raise RuntimeError(
                f"Kolekcja {name!r} ma wektory o wymiarze {size}, a model embeddingów zwraca {dim}. "
                "Zmieniono EMBED_MODEL? Uruchom ingest z --recreate."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # TODO(hybrid): sparse_vectors_config={"bm25": models.SparseVectorParams(
        #     modifier=models.Modifier.IDF)}
    )
    # Indeksy payloadu przyspieszają filtrowanie po metadanych (metadata filtering).
    # Tryb lokalny ich nie obsługuje (filtruje bez indeksu), więc tam je pomijamy.
    if client.init_options.get("path") or client.init_options.get("location") == ":memory:":
        return
    for field_name in ("source", "chunking"):
        client.create_payload_index(name, field_name, models.PayloadSchemaType.KEYWORD)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    embed_model: str,
) -> None:
    """Zapisuje chunki z wektorami; tekst i metadane lądują w payloadzie."""
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={DENSE: vector},
            payload={"text": chunk.text, "embed_model": embed_model, **chunk.metadata},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), 256):
        client.upsert(collection_name=collection, points=points[i : i + 256])


def search(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    k: int,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """Wyszukiwanie dense (najbliżsi sąsiedzi w przestrzeni embeddingów)."""
    if not client.collection_exists(collection):
        raise RuntimeError(f"Kolekcja {collection!r} nie istnieje. Uruchom najpierw: make ingest")
    response = client.query_points(
        collection_name=collection,
        query=vector,
        using=DENSE,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )
    return response.points
