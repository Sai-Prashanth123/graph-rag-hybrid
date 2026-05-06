from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document
from config import RAGConfig
from graph_store import GraphStore


def make_config():
    config = RAGConfig()
    config.NEO4J_URI = "neo4j+s://fake.example.com"
    config.NEO4J_USER = "neo4j"
    config.NEO4J_PASSWORD = "fake"
    config.NEO4J_DATABASE = "neo4j"
    return config


@patch("graph_store.GraphDatabase")
def test_init_opens_driver(mock_gdb):
    config = make_config()
    store = GraphStore(config)

    mock_gdb.driver.assert_called_once_with(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    assert store.driver is mock_gdb.driver.return_value


@patch("graph_store.GraphDatabase")
def test_close_closes_driver(mock_gdb):
    config = make_config()
    store = GraphStore(config)
    store.close()
    mock_gdb.driver.return_value.close.assert_called_once()


def test_init_with_empty_uri_raises():
    config = make_config()
    config.NEO4J_URI = ""

    with pytest.raises(ValueError, match="NEO4J_URI"):
        GraphStore(config)


@patch("graph_store.GraphDatabase")
def test_add_chunks_runs_merge_query(mock_gdb):
    config = make_config()
    store = GraphStore(config)

    mock_session = MagicMock()
    mock_gdb.driver.return_value.session.return_value.__enter__.return_value = mock_session

    docs = [
        Document(
            page_content="hello world",
            metadata={"chunk_id": "a::0", "name": "a.txt", "chunk_index": 0},
        ),
    ]
    store.add_chunks(docs)

    calls = mock_session.run.call_args_list
    assert any("MERGE (c:Chunk" in str(call) for call in calls)


@patch("graph_store.GraphDatabase")
def test_add_entities_writes_entity_and_mention(mock_gdb):
    config = make_config()
    store = GraphStore(config)

    mock_session = MagicMock()
    mock_gdb.driver.return_value.session.return_value.__enter__.return_value = mock_session

    triples = {
        "chunk_id": "a::0",
        "entities": [
            {"name": "Python", "type": "Technology"},
            {"name": "Alice", "type": "Person"},
        ],
        "relations": [
            {"source": "Alice", "target": "Python", "type": "USES"},
        ],
    }
    store.add_entities(triples)

    calls = [str(c) for c in mock_session.run.call_args_list]
    assert any("MERGE (e:Entity" in c for c in calls)
    assert any("MENTIONED_IN" in c for c in calls)
    assert any("RELATION" in c for c in calls)


@patch("graph_store.GraphDatabase")
def test_search_returns_chunks_for_entities(mock_gdb):
    config = make_config()
    store = GraphStore(config)

    mock_session = MagicMock()
    mock_gdb.driver.return_value.session.return_value.__enter__.return_value = mock_session

    fake_record = MagicMock()
    fake_record.__getitem__.side_effect = lambda key: {
        "id": "a::0",
        "text": "hello",
        "doc_name": "a.txt",
        "chunk_index": 0,
        "relevance": 2,
    }[key]
    mock_session.run.return_value = [fake_record]

    results = store.search(["Python", "Alice"], k=10, hops=1)

    assert len(results) == 1
    assert results[0].page_content == "hello"
    assert results[0].metadata["chunk_id"] == "a::0"


@patch("graph_store.GraphDatabase")
def test_search_with_2_hops_uses_relation_traversal(mock_gdb):
    config = make_config()
    store = GraphStore(config)
    mock_session = MagicMock()
    mock_gdb.driver.return_value.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = []

    store.search(["Python"], k=10, hops=2)

    call_query = str(mock_session.run.call_args_list[0])
    assert "RELATION" in call_query


def test_search_with_no_entities_returns_empty():
    with patch("graph_store.GraphDatabase"):
        config = make_config()
        store = GraphStore(config)
        results = store.search([], k=10, hops=1)
    assert results == []
