from unittest.mock import MagicMock
from langchain_core.documents import Document
from config import RAGConfig
from hybrid_retriever import HybridRetriever


def doc(cid: str) -> Document:
    return Document(page_content=f"text-{cid}", metadata={"chunk_id": cid})


def make_config():
    config = RAGConfig()
    config.RETRIEVER_K = 5
    config.TOP_K = 3
    config.RRF_K = 60
    config.GRAPH_HOPS = 1
    return config


def test_invoke_runs_all_three_and_fuses():
    config = make_config()

    vec = MagicMock()
    vec.invoke = MagicMock(return_value=[doc("a"), doc("b"), doc("c")])

    bm25 = MagicMock()
    bm25.search = MagicMock(return_value=[doc("a"), doc("d"), doc("e")])

    graph = MagicMock()
    graph.search = MagicMock(return_value=[doc("a"), doc("f"), doc("g")])

    extractor = MagicMock()
    extractor._extract_entity_names = MagicMock(return_value=["X"])

    retriever = HybridRetriever(
        vector_retriever=vec,
        bm25_retriever=bm25,
        graph_retriever=graph,
        graph_extractor=extractor,
        config=config,
    )
    result = retriever.invoke("question")

    vec.invoke.assert_called_once()
    bm25.search.assert_called_once()
    graph.search.assert_called_once()

    assert len(result) <= config.TOP_K
    assert any(d.metadata["chunk_id"] == "a" for d in result)


def test_invoke_skips_disabled_retrievers():
    config = make_config()
    vec = MagicMock()
    vec.invoke = MagicMock(return_value=[doc("a"), doc("b")])

    retriever = HybridRetriever(
        vector_retriever=vec,
        bm25_retriever=None,
        graph_retriever=None,
        graph_extractor=None,
        config=config,
    )
    result = retriever.invoke("question")
    assert len(result) > 0
    assert any(d.metadata["chunk_id"] == "a" for d in result)


def test_invoke_handles_graph_failure_gracefully():
    config = make_config()
    vec = MagicMock()
    vec.invoke = MagicMock(return_value=[doc("a")])
    bm25 = MagicMock()
    bm25.search = MagicMock(return_value=[doc("b")])

    graph = MagicMock()
    graph.search = MagicMock(side_effect=Exception("Neo4j down"))
    extractor = MagicMock()
    extractor._extract_entity_names = MagicMock(return_value=["X"])

    retriever = HybridRetriever(
        vector_retriever=vec,
        bm25_retriever=bm25,
        graph_retriever=graph,
        graph_extractor=extractor,
        config=config,
    )
    result = retriever.invoke("question")
    assert len(result) > 0
