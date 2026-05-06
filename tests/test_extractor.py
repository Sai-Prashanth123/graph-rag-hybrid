import json
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from config import RAGConfig
from graph_extractor import GraphExtractor


def make_extractor():
    config = RAGConfig()
    config.EXTRACTION_MAX_CHARS = 100
    config.EXTRACTION_CONCURRENCY = 2
    fake_llm = MagicMock()
    return GraphExtractor(fake_llm, config), fake_llm


def test_parse_valid_json():
    extractor, _ = make_extractor()
    raw = json.dumps({
        "entities": [{"name": "Python", "type": "Technology"}],
        "relations": [{"source": "Alice", "target": "Python", "type": "USES"}],
    })
    parsed = extractor._parse(raw)
    assert parsed["entities"][0]["name"] == "Python"
    assert parsed["relations"][0]["type"] == "USES"


def test_parse_strips_codeblock_wrapper():
    extractor, _ = make_extractor()
    raw = '```json\n{"entities":[],"relations":[]}\n```'
    parsed = extractor._parse(raw)
    assert parsed["entities"] == []


def test_parse_malformed_returns_empty():
    extractor, _ = make_extractor()
    parsed = extractor._parse("not json at all")
    assert parsed == {"entities": [], "relations": []}


def test_truncate_respects_max_chars():
    extractor, _ = make_extractor()
    text = "x" * 500
    truncated = extractor._truncate(text)
    assert len(truncated) == 100


@pytest.mark.asyncio
async def test_extract_and_store_calls_graph_for_each_doc():
    config = RAGConfig()
    config.EXTRACTION_MAX_CHARS = 1000
    config.EXTRACTION_CONCURRENCY = 2

    fake_llm = MagicMock()
    async def fake_ainvoke(_messages):
        resp = MagicMock()
        resp.content = '{"entities":[{"name":"X","type":"Concept"}],"relations":[]}'
        return resp
    fake_llm.ainvoke = fake_ainvoke

    extractor = GraphExtractor(fake_llm, config)

    fake_graph = MagicMock()
    docs = [
        Document(page_content="text 1", metadata={"chunk_id": "a::0"}),
        Document(page_content="text 2", metadata={"chunk_id": "a::1"}),
        Document(page_content="text 3", metadata={"chunk_id": "a::2"}),
    ]

    await extractor.extract_and_store_async(docs, fake_graph)

    assert fake_graph.add_entities.call_count == 3
    chunk_ids = {
        call.args[0]["chunk_id"]
        for call in fake_graph.add_entities.call_args_list
    }
    assert chunk_ids == {"a::0", "a::1", "a::2"}
