"""Ingest: wczytanie plików PDF/MD/TXT → normalizacja → chunking → embedding → Qdrant.

Każdy loader zwraca listę `Block` – fragmentów dokumentu z metadanymi (strona dla PDF,
sekcja dla Markdowna). Chunking dzieli bloki dalej, zachowując ich metadane.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag.chunking import Block, Chunk, chunk_blocks
from rag.config import Settings
from rag.embeddings import embed_documents
from rag.normalize import basic_clean, join_split_words, word_counts
from rag.store import ensure_collection, get_client, upsert_chunks

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".md", ".txt")


# --------------------------------------------------------------------------- loaders


def load_pdf(path: Path) -> list[Block]:
    """Jeden blok na stronę; `page` liczymy od 1 (tak jak w czytniku PDF)."""
    reader = PdfReader(path)
    return [
        Block(text=page.extract_text() or "", metadata={"source": path.name, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def load_markdown(path: Path) -> list[Block]:
    """Jeden blok na sekcję; `section` = tekst najbliższego nagłówka."""
    text = path.read_text(encoding="utf-8")
    blocks: list[Block] = []
    matches = list(_MD_HEADING.finditer(text))
    if not matches or matches[0].start() > 0:
        intro = text[: matches[0].start()] if matches else text
        if intro.strip():
            blocks.append(Block(intro, {"source": path.name, "section": ""}))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end]
        blocks.append(Block(body, {"source": path.name, "section": m.group(2)}))
    return blocks


def load_text(path: Path) -> list[Block]:
    """Cały plik TXT jako jeden blok."""
    return [Block(path.read_text(encoding="utf-8"), {"source": path.name})]


def load_file(path: Path) -> list[Block]:
    """Wybiera loader po rozszerzeniu pliku."""
    loaders = {".pdf": load_pdf, ".md": load_markdown, ".txt": load_text}
    return loaders[path.suffix.lower()](path)


def discover_files(paths: list[Path]) -> list[Path]:
    """Rozwija katalogi do listy obsługiwanych plików (rekurencyjnie, posortowane)."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
        elif p.exists():
            files.append(p)
        else:
            log.warning("Pomijam brakujący plik: %s", p)
    return files


# --------------------------------------------------------------------------- pipeline


def source_ids_for(metadata: dict[str, Any]) -> list[str]:
    """Stabilne identyfikatory źródła, NIEZALEŻNE od strategii chunkingu.

    Złoty zbiór (eval/) odwołuje się do nich, więc recall@k można porównywać między
    strategiami chunkingu. Format: `plik.pdf#p3`, `plik.md#Nazwa sekcji`, `plik.txt`.
    """
    source = metadata["source"]
    if "page" in metadata:
        return [f"{source}#p{metadata['page']}"]
    if metadata.get("section"):
        return [f"{source}#{metadata['section']}"]
    return [source]


def normalize_blocks(blocks: list[Block]) -> list[Block]:
    """Normalizuje tekst bloków; słownik do sklejania słów liczymy z całego dokumentu."""
    cleaned = [basic_clean(b.text) for b in blocks]
    vocab: Counter[str] = word_counts("\n".join(cleaned))
    return [
        Block(join_split_words(text, vocab), b.metadata)
        for text, b in zip(cleaned, blocks, strict=True)
        if text
    ]


def build_chunks(paths: list[Path], settings: Settings, strategy: str | None = None) -> list[Chunk]:
    """Wczytuje pliki i zwraca chunki gotowe do embeddingu (bez wywołań sieciowych)."""
    strategy = strategy or settings.chunking
    chunks: list[Chunk] = []
    for path in discover_files(paths):
        blocks = normalize_blocks(load_file(path))
        for b in blocks:
            b.metadata["source_ids"] = source_ids_for(b.metadata)
        file_chunks = chunk_blocks(
            blocks, strategy, size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        log.info(
            "%s: %d bloków → %d chunków (%s)", path.name, len(blocks), len(file_chunks), strategy
        )
        chunks += file_chunks
    return chunks


def ingest(
    paths: list[Path],
    settings: Settings,
    strategy: str | None = None,
    collection: str | None = None,
    recreate: bool = False,
) -> dict[str, Any]:
    """Pełny ingest. Zwraca statystyki (liczba chunków, czasy etapów)."""
    collection = collection or settings.collection
    t0 = time.perf_counter()
    chunks = build_chunks(paths, settings, strategy)
    if not chunks:
        raise RuntimeError(f"Brak dokumentów do zaindeksowania w: {[str(p) for p in paths]}")
    t1 = time.perf_counter()
    vectors = embed_documents([c.text for c in chunks], settings)
    t2 = time.perf_counter()
    client = get_client(settings.qdrant_url)
    ensure_collection(client, collection, dim=len(vectors[0]), recreate=recreate)
    upsert_chunks(client, collection, chunks, vectors, embed_model=settings.embed_model)
    t3 = time.perf_counter()
    return {
        "collection": collection,
        "chunks": len(chunks),
        "files": len({c.metadata["source"] for c in chunks}),
        "embed_s": round(t2 - t1, 2),
        "total_s": round(t3 - t0, 2),
    }
