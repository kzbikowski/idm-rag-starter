import pytest

from rag.eval import recall_at_k, reciprocal_rank


def test_recall_at_k():
    retrieved = [["a"], ["b", "c"], ["d"]]
    assert recall_at_k(retrieved, ["a", "d"], k=1) == 0.5
    assert recall_at_k(retrieved, ["a", "d"], k=3) == 1.0
    assert recall_at_k(retrieved, ["c"], k=2) == 1.0


def test_reciprocal_rank():
    retrieved = [["x"], ["y"], ["a"]]
    assert reciprocal_rank(retrieved, ["a"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["zzz"]) == 0.0
    assert reciprocal_rank(retrieved, ["y", "a"]) == 0.5


def test_to_level_paragraph():
    from rag.eval import to_level

    ids = ["r.pdf#§36.2", "r.pdf#§36", "r.pdf#§3a.1", "p.pdf#pkt2a"]
    assert to_level(ids, "paragraph") == ["r.pdf#§36", "r.pdf#§3a", "p.pdf#pkt2a"]
    assert to_level(ids, "source") == ids
