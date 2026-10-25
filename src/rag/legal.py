"""Dokumenty o strukturze prawnej (regulaminy, przepisy): rozdział → § → ustęp.

Przepływ dla jednego dokumentu:
1. `build_stream`  – strony PDF → jeden ciągły tekst bez numerów stron, stopek i nagłówków
   bieżących; strony tytułowe rozdziałów zamieniamy na linię „Rozdział III. Przyjęcie na studia”.
   Zapamiętujemy, od którego znaku zaczyna się każda strona (żeby znać stronę chunka).
2. `parse_units` (woła go `build_stream`) – dzielimy tekst na jednostki (ustępy) i zapisujemy
   ich zakres znaków oraz metadane: rozdział, paragraf, ustęp, strona.
3. Chunking:
   - `structure_chunks` – jeden chunk = jeden ustęp (+ nagłówek z kontekstem: rozdział, §);
   - `span_chunks`      – dowolny podział tekstu (fixed-size, recursive); `source_ids` chunka
     wyznaczamy z jednostek, na które chunk zachodzi. Dzięki temu obie strategie można
     uczciwie porównać tymi samymi metrykami (recall@k po paragrafach).

Format jest dopasowany do Regulaminu studiów PW, ale reguły są proste
i opisane – łatwo je dostosować do innego aktu.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any

from rag.chunking import Block, Chunk, recursive_split

LEGAL_DOC_TYPES = ("regulamin", "przepisy_przejsciowe")

_ROMAN = r"[IVXL]+"
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_FOOTER = re.compile(r"^Regulamin\s+studiów\s+w\s+PW\s*\|.*$", re.MULTILINE)
_ROMAN_LINE = re.compile(rf"^\s*({_ROMAN})\s*$")
_CHAPTER = re.compile(rf"^Rozdział ({_ROMAN})\. (.+)$")
# „§ 12” w osobnej linii (nie „§ 22 ust. 2” zawinięte z poprzedniej linii).
_PARAGRAPH = re.compile(r"^§\s*(\d+[a-z]?)\s*$")
# „3. Treść…” albo „3a. Treść…” na początku linii.
_USTEP = re.compile(r"^(\d+[a-z]?)\.\s+\S")
_REPEALED = re.compile(r"^\s*\d+[a-z]?\.\s*\(uchylony\)\s*$")
_FOOTNOTE = re.compile(r"^\*")  # przypis, np. „*Zob.: § 2 Uchwały nr …”


@dataclass
class Unit:
    """Jednostka redakcyjna aktu: ustęp (albo cały §, gdy nie ma numerowanych ustępów)."""

    start: int
    end: int
    paragraph: str | None  # "12"; None dla dokumentów bez § (np. przepisy przejściowe)
    ustep: str | None  # "3a"; None, gdy § nie dzieli się na ustępy
    kind: str = "ustep"  # "ustep" | "wstep" (tekst przed pierwszą jednostką) | "przypis"
    chapter: str | None = None  # "III Przyjęcie na studia"
    page: int | None = None


@dataclass
class LegalDoc:
    """Ciągły tekst dokumentu + mapa stron + jednostki."""

    text: str
    page_starts: list[int]  # offset początku kolejnych stron w `text`
    pages: list[int]  # numery stron odpowiadające `page_starts`
    units: list[Unit] = field(default_factory=list)

    def page_at(self, offset: int) -> int:
        i = bisect.bisect_right(self.page_starts, offset) - 1
        return self.pages[max(i, 0)]


# --------------------------------------------------------------------------- 1. tekst


def _chapter_title_page(text: str) -> tuple[str, str] | None:
    """Strona tytułowa rozdziału: kilka linii tytułu i linia z samą liczbą rzymską."""
    if "§" in text:
        return None
    lines = [ln.strip() for ln in _FOOTER.sub("", text).splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if _ROMAN_LINE.match(line) and i > 0:
            return line, " ".join(lines[:i])
    return None


def _strip_running_header(text: str, roman: str, title: str) -> str:
    """Usuwa nagłówek bieżący strony, np. „III Przyjęcie na studia” (może być w 2 liniach)."""
    words = r"\s+".join(re.escape(w) for w in title.split())
    return re.sub(rf"^\s*{roman}\s+{words}\s*\n", "", text, count=1)


def build_stream(blocks: list[Block]) -> LegalDoc:
    """Skleja strony w jeden tekst, usuwając elementy layoutu (patrz docstring modułu)."""
    has_chapter_pages = any(_chapter_title_page(b.text) for b in blocks)
    parts: list[str] = []
    page_starts: list[int] = []
    pages: list[int] = []
    offset = 0
    chapter: tuple[str, str] | None = None
    for block in blocks:
        lines = block.text.splitlines()
        if lines and _PAGE_NUMBER.match(lines[0]):
            lines = lines[1:]  # numer strony w pierwszej linii
        text = _FOOTER.sub("", "\n".join(lines)).strip()

        if title_page := _chapter_title_page(text):
            chapter = title_page
            text = f"Rozdział {chapter[0]}. {chapter[1]}"
        elif has_chapter_pages and chapter is None:
            continue  # okładka i spis treści przed pierwszym rozdziałem
        elif chapter:
            text = _strip_running_header(text, *chapter)
        if not text:
            continue
        page_starts.append(offset)
        pages.append(block.metadata.get("page", 1))
        parts.append(text)
        offset += len(text) + 1  # +1 za "\n" łączący strony
    doc = LegalDoc(text="\n".join(parts), page_starts=page_starts, pages=pages)
    parse_units(doc)
    return doc


# --------------------------------------------------------------------------- 2. jednostki


def _number(label: str) -> int:
    return int(re.match(r"\d+", label).group())


def parse_units(doc: LegalDoc) -> list[Unit]:
    """Dzieli tekst na ustępy. Numeracja musi rosnąć, co odsiewa fałszywe trafienia
    (np. linia zaczynająca się od „2. ” w środku wyliczenia z innego miejsca)."""
    units: list[Unit] = []
    chapter: str | None = None
    paragraph: str | None = None
    ustep: str | None = None
    offset = 0

    def open_unit(start: int, kind: str = "ustep") -> None:
        if units:
            units[-1].end = start
        units.append(Unit(start, len(doc.text), paragraph, ustep, kind, chapter))

    open_unit(0, kind="wstep")  # tekst przed pierwszym § / ustępem (np. wstęp przepisów)
    for line in doc.text.split("\n"):
        stripped = line.strip()
        if _FOOTNOTE.match(stripped):
            paragraph = ustep = None
            open_unit(offset, kind="przypis")
        elif m := _CHAPTER.match(stripped):
            chapter = f"{m.group(1)} {m.group(2)}"
        elif (m := _PARAGRAPH.match(stripped)) and (
            paragraph is None or _number(m.group(1)) > _number(paragraph)
        ):
            paragraph, ustep = m.group(1), None
            open_unit(offset)
        elif (m := _USTEP.match(stripped)) and _is_next_ustep(m.group(1), ustep):
            ustep = m.group(1)
            open_unit(offset)
        offset += len(line) + 1

    for unit in units:
        unit.page = doc.page_at(unit.start)
    doc.units = units
    return units


def _is_next_ustep(candidate: str, current: str | None) -> bool:
    """„1.” zaczyna numerację; kolejne numery mogą rosnąć o 1–2 (lub mieć literę: 3 → 3a)."""
    n = _number(candidate)
    if current is None:
        return n == 1
    return _number(current) < n <= _number(current) + 2 or (
        n == _number(current) and candidate != current
    )


# --------------------------------------------------------------------------- 3. chunki


def citation(metadata: dict[str, Any]) -> str | None:
    """Cytowanie w formie prawniczej: „§ 12 ust. 3”, „§ 3”, „pkt 2a” (przepisy przejściowe)."""
    if metadata.get("unit_kind") in ("wstep", "przypis"):
        return metadata["unit_kind"].replace("wstep", "wstęp")
    paragraph, ustep = metadata.get("paragraph"), metadata.get("ustep")
    if paragraph:
        return f"§ {paragraph}" + (f" ust. {ustep}" if ustep else "")
    if ustep:
        return f"pkt {ustep}"
    return None


def unit_source_ids(source: str, unit: Unit) -> list[str]:
    """Identyfikatory źródła: `plik#§12` i `plik#§12.3` (przepisy bez §: `plik#pkt2a`)."""
    if unit.kind != "ustep":
        return [f"{source}#{unit.kind}"]
    if unit.paragraph is None:
        return [f"{source}#pkt{unit.ustep}"] if unit.ustep else [source]
    ids = [f"{source}#§{unit.paragraph}"]
    if unit.ustep:
        ids.append(f"{source}#§{unit.paragraph}.{unit.ustep}")
    return ids


def _unit_metadata(unit: Unit) -> dict[str, Any]:
    return {
        "chapter": unit.chapter,
        "paragraph": unit.paragraph,
        "ustep": unit.ustep,
        "unit_kind": unit.kind,
        "page": unit.page,
    }


def unit_body(doc: LegalDoc, unit: Unit) -> str:
    """Treść jednostki bez linii technicznych: „Rozdział …” i samego nagłówka „§ 12”."""
    lines = doc.text[unit.start : unit.end].split("\n")
    kept = [
        ln for ln in lines if not _CHAPTER.match(ln.strip()) and not _PARAGRAPH.match(ln.strip())
    ]
    return "\n".join(kept).strip()


def structure_chunks(doc: LegalDoc, meta: dict[str, Any], max_chars: int) -> list[Chunk]:
    """Chunk = ustęp. Każdy chunk zaczyna się nagłówkiem z kontekstem, np.
    „Regulamin studiów PW (od 1.10.2026) › Rozdział III Przyjęcie na studia › § 12 ust. 3”,
    bo sam tekst ustępu („3. Zasady… określają…”) nie mówi, czego dotyczy."""
    chunks: list[Chunk] = []
    for unit in doc.units:
        body = unit_body(doc, unit)
        if len(body) < 20 or _REPEALED.match(body):
            continue  # ustępy „(uchylony)” nie niosą treści
        unit_meta = _unit_metadata(unit)
        cite = citation(unit_meta)
        header = " › ".join(
            p
            for p in (
                meta.get("doc_label"),
                f"Rozdział {unit.chapter}" if unit.chapter else None,
                cite,
            )
            if p
        )
        # Długie ustępy (np. słowniczek pojęć) dzielimy po liniach/punktach.
        pieces = (
            recursive_split(body, max_chars, 0, ("\n", ". ", " "))
            if len(body) > max_chars
            else [body]
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=f"{header}\n{piece}",
                    metadata={
                        **meta,
                        **unit_meta,
                        "source_ids": unit_source_ids(meta["source"], unit),
                        "chunking": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks


MIN_OVERLAP = 100  # znaków – mniejsze zahaczenie o ustęp nie liczy się jako „zawiera ten ustęp”


def span_chunks(
    doc: LegalDoc, meta: dict[str, Any], texts: list[str], strategy: str
) -> list[Chunk]:
    """Chunki z dowolnego splittera (fixed/recursive) + `source_ids` z jednostek, na które
    chunk zachodzi o co najmniej `MIN_OVERLAP` znaków (albo pół krótkiej jednostki)."""
    chunks: list[Chunk] = []
    cursor = 0
    for text in texts:
        start = doc.text.find(text[:40], cursor)
        if start == -1:
            start = cursor
        end = start + len(text)
        cursor = start + 1
        ids: list[str] = []
        for unit in doc.units:
            overlap = min(end, unit.end) - max(start, unit.start)
            if overlap >= min(MIN_OVERLAP, (unit.end - unit.start) // 2) and overlap > 0:
                ids += [i for i in unit_source_ids(meta["source"], unit) if i not in ids]
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    **meta,
                    "page": doc.page_at(start),
                    "source_ids": ids or [meta["source"]],
                    "chunking": strategy,
                    "chunk_index": len(chunks),
                },
            )
        )
    return chunks
