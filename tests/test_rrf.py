from langchain_core.documents import Document
from hybrid_retriever import rrf_fuse


def doc(chunk_id: str) -> Document:
    return Document(page_content=f"text-{chunk_id}", metadata={"chunk_id": chunk_id})


def test_rrf_fuses_three_rankings():
    r1 = [doc("a"), doc("b"), doc("c")]
    r2 = [doc("a"), doc("c"), doc("d")]
    r3 = [doc("a"), doc("e"), doc("f")]

    fused = rrf_fuse([r1, r2, r3], k=60)

    assert fused[0].metadata["chunk_id"] == "a"


def test_rrf_handles_empty_rankings():
    r1 = [doc("x"), doc("y")]
    r2 = []
    r3 = [doc("y"), doc("x")]

    fused = rrf_fuse([r1, r2, r3], k=60)

    assert len(fused) == 2
    ids = [d.metadata["chunk_id"] for d in fused]
    assert "x" in ids and "y" in ids


def test_rrf_dedupes_by_chunk_id():
    r1 = [doc("a"), doc("b")]
    r2 = [doc("a"), doc("b")]

    fused = rrf_fuse([r1, r2], k=60)

    assert len(fused) == 2
    assert {d.metadata["chunk_id"] for d in fused} == {"a", "b"}


def test_rrf_consistently_top_doc_wins():
    # 'a' is rank 0 in both rankings — must win.
    r1 = [doc("a"), doc("b"), doc("c"), doc("d")]
    r2 = [doc("a"), doc("c"), doc("b"), doc("d")]

    fused = rrf_fuse([r1, r2], k=60)
    ids = [d.metadata["chunk_id"] for d in fused]

    assert ids[0] == "a"
    # 'd' is last in both — must be last
    assert ids[-1] == "d"


def test_rrf_returns_empty_for_all_empty():
    fused = rrf_fuse([[], [], []], k=60)
    assert fused == []
