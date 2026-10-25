import pytest

from rag.generate import Answer, RagResult, best_quote, inline_markers
from rag.retrieve import Hit


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("[2], [3], [4]", [2, 3, 4]),
        ("2, 3", [2, 3]),
        (["[1]", "3"], [1, 3]),
        ([1, 2], [1, 2]),
        (4, [4]),
        (None, []),
        ("", []),
    ],
)
def test_repair_citations(raw, expected):
    assert Answer.model_validate({"answer": "x", "citations": raw}).citations == expected


def test_repair_from_json_string_like_qwen_returns():
    raw = '{"answer": "Tekst [2].", "citations": "[2], [3]"}'
    assert Answer.model_validate_json(raw).citations == [2, 3]


def test_inline_markers():
    assert inline_markers("A [2]. B [1], [2]. C [10].") == [2, 1, 10]


def _result(answer: str, citations: list[int], n_hits: int = 3) -> RagResult:
    hits = [
        Hit(text=f"chunk {i}", score=0.5, metadata={"source": f"f{i}.md"}) for i in range(n_hits)
    ]
    return RagResult("q", Answer(answer=answer, citations=citations), hits, "tool", 0.1, 1.0)


def test_cited_numbers_merge_inline_and_list_and_drop_invalid():
    r = _result("Coś [1]. Coś innego [7].", citations=[3])
    assert r.cited_numbers == [1, 3]
    assert r.has_inline_citations
    assert [i for i, _ in r.retrieved_only()] == [2]


def test_refusal_is_empty_citations():
    r = _result("W dokumentach nie ma tej informacji.", citations=[])
    assert not r.answered_from_context
    assert not r.has_inline_citations


def test_best_quote_picks_overlapping_sentence():
    text = "Qdrant to baza wektorowa. Okno kontekstu ustawiamy na 8192 tokeny. Koniec."
    assert best_quote(text, "Okno kontekstu powinno mieć 8192 tokeny [1].") == (
        "Okno kontekstu ustawiamy na 8192 tokeny."
    )


def test_best_quote_joins_broken_lines_and_skips_headings():
    text = "## Fixed-size\n\nChunking fixed-size tnie tekst na kawałki o stałej\nliczbie znaków. Inne zdanie."
    assert best_quote(text, "fixed-size tnie tekst na kawałki [1]") == (
        "Chunking fixed-size tnie tekst na kawałki o stałej liczbie znaków."
    )
