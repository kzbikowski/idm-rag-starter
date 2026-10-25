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
