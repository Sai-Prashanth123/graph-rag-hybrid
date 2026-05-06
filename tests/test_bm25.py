import os
import tempfile
from langchain_core.documents import Document
from config import RAGConfig
from bm25_retriever import BM25Retriever


def make_config(tmp_path):
    config = RAGConfig()
    config.BM25_INDEX_PATH = str(tmp_path / "bm25.pkl")
    return config


def test_add_and_retrieve(tmp_path):
    config = make_config(tmp_path)
    bm25 = BM25Retriever(config)

    docs = [
        Document(page_content="Python is a programming language",
                 metadata={"chunk_id": "a::0"}),
        Document(page_content="Neo4j is a graph database",
                 metadata={"chunk_id": "b::0"}),
        Document(page_content="ChromaDB stores vector embeddings",
                 metadata={"chunk_id": "c::0"}),
    ]
    bm25.add_documents(docs)

    results = bm25.search("Python language", k=2)

    assert len(results) == 2
    assert results[0].metadata["chunk_id"] == "a::0"


def test_persist_and_reload(tmp_path):
    config = make_config(tmp_path)

    bm25_a = BM25Retriever(config)
    bm25_a.add_documents([
        Document(page_content="The quick brown fox",
                 metadata={"chunk_id": "x::0"}),
    ])
    bm25_a.save()

    bm25_b = BM25Retriever(config)
    bm25_b.load()

    results = bm25_b.search("fox", k=1)
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "x::0"


def test_empty_index_returns_empty_list(tmp_path):
    config = make_config(tmp_path)
    bm25 = BM25Retriever(config)

    results = bm25.search("anything", k=5)
    assert results == []
