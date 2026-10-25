"""Chunking: podział tekstu na fragmenty (chunki) do embeddingu.

Dwie strategie:
- `fixed`: okno o stałej długości (w znakach) z overlapem – najprostsze, ignoruje strukturę;
- `recursive`: dzieli kolejno po akapitach, liniach, zdaniach, słowach, aż fragment zmieści się
  w limicie, a potem skleja małe kawałki – zachowuje naturalne granice tekstu.

Długość mierzymy w znakach, nie w tokenach (prościej; dla polskiego ~1 token ≈ 3–4 znaki).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass
class Block:
    """Spójny fragment dokumentu przed chunkingiem (strona PDF, sekcja Markdowna, cały plik TXT)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Fragment tekstu z metadanymi, który trafi do vector store."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministyczne id (uuid5) → ponowny ingest nadpisuje punkty zamiast je dublować."""
        key = f"{self.metadata.get('source')}|{self.metadata.get('chunking')}|{self.metadata.get('chunk_index')}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def fixed_size(text: str, size: int, overlap: int) -> list[str]:
    """Okno przesuwne: chunki po `size` znaków, kolejne zachodzą na siebie o `overlap`."""
    step = size - overlap  # overlap < size gwarantuje walidacja w config.py
    chunks = [text[i : i + size] for i in range(0, len(text), step)]
    # Ostatni chunk bywa w całości zawarty w overlapie poprzedniego – wtedy go pomijamy.
    if len(chunks) > 1 and len(chunks[-1]) <= overlap:
        chunks.pop()
    return [c.strip() for c in chunks if c.strip()]


def recursive_split(
    text: str, size: int, overlap: int, separators: tuple[str, ...] = DEFAULT_SEPARATORS
) -> list[str]:
    """Rekurencyjny podział po coraz drobniejszych separatorach.

    1. Dzielimy tekst pierwszym separatorem (np. akapity).
    2. Kawałki dłuższe niż `size` dzielimy rekurencyjnie kolejnym separatorem.
    3. Sąsiednie krótkie kawałki sklejamy, dopóki mieszczą się w `size`;
       nowy chunk zaczynamy od końcówki poprzedniego (ok. `overlap` znaków).
    """
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    if not separators:
        return fixed_size(text, size, overlap)

    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for part in text.split(sep):
        if len(part) > size:
            pieces += recursive_split(part, size, overlap, rest)
        elif part.strip():
            pieces.append(part)
    return _merge(pieces, sep, size, overlap)


def _merge(pieces: list[str], sep: str, size: int, overlap: int) -> list[str]:
    """Skleja kawałki w chunki ≤ `size`, z overlapem złożonym z całych kawałków."""
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len(sep.join([*current, piece])) > size:
            chunks.append(sep.join(current).strip())
            # Overlap: zostawiamy końcowe kawałki, których łączna długość ≤ overlap.
            while current and len(sep.join(current)) > overlap:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(sep.join(current).strip())
    return [c for c in chunks if c]


def chunk_blocks(blocks: list[Block], strategy: str, size: int, overlap: int) -> list[Chunk]:
    """Dzieli bloki dokumentu na chunki; każdy chunk dziedziczy metadane bloku."""
    split = {"fixed": fixed_size, "recursive": recursive_split}[strategy]

    chunks: list[Chunk] = []
    for block in blocks:
        for text in split(block.text, size, overlap):
            meta = {**block.metadata, "chunking": strategy, "chunk_index": len(chunks)}
            chunks.append(Chunk(text=text, metadata=meta))
    return chunks
