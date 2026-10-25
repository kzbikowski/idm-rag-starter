"""Parser struktury prawnej na syntetycznych stronach."""

from rag.chunking import Block
from rag.legal import build_stream, citation, span_chunks, structure_chunks

META = {"source": "reg.pdf", "doc_label": "Regulamin", "doc_type": "regulamin"}

PAGES_2026 = [
    "1\nRegulamin\nstudiów",  # okładka
    "2\nI Postanowienia ogólne\n → \nSpis rzeczy",  # spis treści
    "3\nPostanowienia ogólne\nI\nRegulamin\tstudiów\tw\t\tPW   | ",  # strona tytułowa rozdziału
    "4\nI Postanowienia ogólne\n  § 1\n1. Regulamin określa organizację studiów.\n"
    "2. Przepisy stosuje się do wszystkich jednostek.\n"
    "Regulamin\tstudiów\tw\t\tPW    |   Postanowienia ogólne",
    "5\nI Postanowienia ogólne\n§ 2\nStudentów reprezentuje samorząd studencki w sprawach studentów.\n"
    "§ 3\n1. Zwierzchnikiem studentów jest Rektor.\n2. (uchylony)\n2a. Rektor może upoważnić inną osobę.\n"
    "3. Przepis § 1 ust. 2 stosuje się odpowiednio.",
]


def _doc(pages):
    blocks = [Block(t, {"source": "reg.pdf", "page": i}) for i, t in enumerate(pages, start=1)]
    return build_stream(blocks)


def test_layout_is_removed_and_chapter_line_inserted():
    doc = _doc(PAGES_2026)
    assert doc.text.startswith("Rozdział I. Postanowienia ogólne")
    assert "Spis rzeczy" not in doc.text and "|" not in doc.text
    assert "\nI Postanowienia ogólne" not in doc.text  # nagłówek bieżący usunięty


def test_units_paragraphs_ustepy_and_pages():
    doc = _doc(PAGES_2026)
    units = [(u.paragraph, u.ustep, u.page) for u in doc.units if u.kind == "ustep" and u.ustep]
    assert units == [
        ("1", "1", 4),
        ("1", "2", 4),
        ("3", "1", 5),
        ("3", "2", 5),
        ("3", "2a", 5),
        ("3", "3", 5),
    ]
    assert all(u.chapter == "I Postanowienia ogólne" for u in doc.units if u.paragraph)


def test_structure_chunks_skip_repealed_and_have_context_header():
    chunks = structure_chunks(_doc(PAGES_2026), META, max_chars=1000)
    ids = [c.metadata["source_ids"][-1] for c in chunks]
    assert ids == [
        "reg.pdf#§1.1",
        "reg.pdf#§1.2",
        "reg.pdf#§2",
        "reg.pdf#§3.1",
        "reg.pdf#§3.2a",
        "reg.pdf#§3.3",
    ]
    assert chunks[0].text.startswith("Regulamin › Rozdział I Postanowienia ogólne › § 1 ust. 1\n")
    # § bez ustępów jest jedną jednostką, a nagłówek „§ 2” nie trafia do treści
    assert "§ 2\n" not in chunks[2].text.split("\n", 1)[1]


def test_reference_inside_text_does_not_start_new_paragraph():
    doc = _doc(PAGES_2026)
    assert [u.paragraph for u in doc.units if u.paragraph] == [
        "1",
        "1",
        "1",
        "2",
        "3",
        "3",
        "3",
        "3",
        "3",
    ]


def test_span_chunks_get_ids_of_overlapped_units():
    doc = _doc(PAGES_2026)
    whole = span_chunks(doc, META, [doc.text], "fixed")[0]
    assert {"reg.pdf#§1.1", "reg.pdf#§3.2a"} <= set(whole.metadata["source_ids"])


def test_citation_formats():
    assert citation({"paragraph": "12", "ustep": "3"}) == "§ 12 ust. 3"
    assert citation({"paragraph": "5", "ustep": None}) == "§ 5"
    assert citation({"paragraph": None, "ustep": "2a"}) == "pkt 2a"
    assert citation({"unit_kind": "wstep"}) == "wstęp"
    assert citation({"source": "x.md"}) is None
