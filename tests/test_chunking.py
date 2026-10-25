from rag.chunking import Block, Chunk, chunk_blocks, fixed_size, recursive_split


def test_fixed_size_respects_size_and_overlap():
    text = "".join(str(i % 10) for i in range(100))
    chunks = fixed_size(text, size=30, overlap=10)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0][-10:] == chunks[1][:10]  # overlap
    assert chunks[-1].endswith(text[-5:])  # nic nie zgubiono na końcu


def test_recursive_prefers_paragraph_boundaries():
    text = "Akapit pierwszy.\n\nAkapit drugi, nieco dłuższy.\n\nAkapit trzeci."
    chunks = recursive_split(text, size=35, overlap=0)
    assert chunks == ["Akapit pierwszy.", "Akapit drugi, nieco dłuższy.", "Akapit trzeci."]


def test_recursive_splits_long_paragraph_by_sentences():
    text = "Zdanie numer jeden. Zdanie numer dwa. Zdanie numer trzy. Zdanie numer cztery."
    chunks = recursive_split(text, size=40, overlap=0)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks).replace(" ", "").replace(".", "") == text.replace(" ", "").replace(
        ".", ""
    )


def test_chunk_blocks_inherits_metadata_and_ids_are_stable():
    blocks = [Block("a " * 100, {"source": "x.txt", "source_ids": ["x.txt"]})]
    chunks = chunk_blocks(blocks, "fixed", size=50, overlap=10)
    assert all(c.metadata["source_ids"] == ["x.txt"] for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    again = chunk_blocks(blocks, "fixed", size=50, overlap=10)
    assert [c.id for c in chunks] == [c.id for c in again]


def test_chunk_id_depends_on_strategy():
    a = Chunk("t", {"source": "s", "chunking": "fixed", "chunk_index": 0})
    b = Chunk("t", {"source": "s", "chunking": "recursive", "chunk_index": 0})
    assert a.id != b.id
