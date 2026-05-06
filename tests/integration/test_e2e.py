import os
import pytest
from langchain_core.documents import Document
from config import RAGConfig
from graph_store import GraphStore

testcontainers_available = True
try:
    from testcontainers.neo4j import Neo4jContainer
except ImportError:
    testcontainers_available = False


def _docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not testcontainers_available
    or not _docker_available()
    or os.getenv("SKIP_INTEGRATION") == "1",
    reason="testcontainers or Docker not available, or SKIP_INTEGRATION=1",
)


@pytest.fixture(scope="module")
def neo4j_container():
    with Neo4jContainer("neo4j:5") as container:
        yield container


@pytest.fixture
def graph_store(neo4j_container):
    config = RAGConfig()
    config.NEO4J_URI = neo4j_container.get_connection_url()
    config.NEO4J_USER = "neo4j"
    config.NEO4J_PASSWORD = neo4j_container.password
    config.NEO4J_DATABASE = "neo4j"
    store = GraphStore(config)
    yield store
    with store.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    store.close()


def test_add_chunks_and_entities_round_trip(graph_store):
    chunk = Document(
        page_content="Alice uses Python at Acme.",
        metadata={"chunk_id": "x::0", "name": "x.txt", "chunk_index": 0},
    )
    graph_store.add_chunks([chunk])

    triples = {
        "chunk_id": "x::0",
        "entities": [
            {"name": "Alice", "type": "Person"},
            {"name": "Python", "type": "Technology"},
            {"name": "Acme", "type": "Organization"},
        ],
        "relations": [
            {"source": "Alice", "target": "Python", "type": "USES"},
            {"source": "Alice", "target": "Acme", "type": "WORKS_AT"},
        ],
    }
    graph_store.add_entities(triples)

    results = graph_store.search(["Alice"], k=10, hops=1)
    assert len(results) >= 1
    assert results[0].metadata["chunk_id"] == "x::0"


def test_graph_dedup_lowercases_entity_name(graph_store):
    chunk = Document(
        page_content="Python is great",
        metadata={"chunk_id": "y::0", "name": "y.txt", "chunk_index": 0},
    )
    graph_store.add_chunks([chunk])
    graph_store.add_entities({
        "chunk_id": "y::0",
        "entities": [{"name": "Python", "type": "Technology"}],
        "relations": [],
    })
    graph_store.add_entities({
        "chunk_id": "y::0",
        "entities": [{"name": "python", "type": "Technology"}],
        "relations": [],
    })

    with graph_store.driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS c")
        count = result.single()["c"]
    assert count == 1, f"Expected 1 entity (deduped); got {count}"


def test_2_hop_traversal_finds_related_chunks(graph_store):
    a = Document(page_content="Alice uses Python.",
                 metadata={"chunk_id": "a::0", "name": "a.txt", "chunk_index": 0})
    b = Document(page_content="Python is fast.",
                 metadata={"chunk_id": "b::0", "name": "b.txt", "chunk_index": 0})
    graph_store.add_chunks([a, b])

    graph_store.add_entities({
        "chunk_id": "a::0",
        "entities": [
            {"name": "Alice", "type": "Person"},
            {"name": "Python", "type": "Technology"},
        ],
        "relations": [{"source": "Alice", "target": "Python", "type": "USES"}],
    })
    graph_store.add_entities({
        "chunk_id": "b::0",
        "entities": [
            {"name": "Python", "type": "Technology"},
            {"name": "Speed", "type": "Concept"},
        ],
        "relations": [{"source": "Python", "target": "Speed", "type": "HAS"}],
    })

    results = graph_store.search(["Speed"], k=10, hops=2)
    chunk_ids = {r.metadata["chunk_id"] for r in results}
    assert "b::0" in chunk_ids
