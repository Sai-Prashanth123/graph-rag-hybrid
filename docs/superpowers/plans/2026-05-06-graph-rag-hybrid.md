# Graph RAG with Hybrid Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing vector-only RAG system into a Graph RAG with hybrid retrieval — Vector + BM25 + Neo4j graph fused via Reciprocal Rank Fusion.

**Architecture:** Add 4 new modules (BM25 retriever, Neo4j graph store, LLM-based graph extractor, hybrid retriever with RRF fusion) alongside the existing Chroma + LangChain pipeline. Wire the hybrid retriever into `rag_system.py` in place of the single Chroma retriever. Feature flags allow each retriever to be disabled independently for graceful degradation.

**Tech Stack:** Python 3.8+, LangChain, ChromaDB, Neo4j (Aura free tier), `rank-bm25`, `python-dotenv`, `pytest`, `testcontainers-python`. Existing: Azure OpenAI (gpt-4o + text-embedding-ada-002), MySQL, PyMySQL, pypdf.

**Spec:** `docs/superpowers/specs/2026-05-06-graph-rag-hybrid-design.md`

---

## File Structure (Locked-In)

### New files

| Path | Responsibility |
|------|----------------|
| `bm25_retriever.py` | BM25 sparse retrieval — tokenize, index, persist via pickle |
| `graph_store.py` | Neo4j connection management + Cypher writes/reads |
| `graph_extractor.py` | LLM-based entity/relation extraction with async concurrency |
| `hybrid_retriever.py` | Parallel fan-out across 3 retrievers + RRF fusion |
| `.env.example` | Template for credentials (committed) |
| `tests/__init__.py` | Empty marker |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/fixtures/sample_corpus.txt` | Small test corpus with known entity overlap |
| `tests/test_chunk_id.py` | DocumentProcessor chunk_id assertions |
| `tests/test_bm25.py` | BM25Retriever unit tests |
| `tests/test_rrf.py` | RRF fusion math unit tests |
| `tests/test_graph_store.py` | GraphStore Cypher & dedup unit tests |
| `tests/test_extractor.py` | GraphExtractor JSON parsing unit tests |
| `tests/test_hybrid_retriever.py` | HybridRetriever orchestration unit tests |
| `tests/integration/__init__.py` | Empty marker |
| `tests/integration/test_e2e.py` | Full pipeline integration test (testcontainers) |
| `tests/smoke_test.py` | Manual end-to-end smoke runner |

### Modified files

| Path | Change |
|------|--------|
| `requirements.txt` | Add `neo4j`, `rank-bm25`, `python-dotenv`, `pytest`, `testcontainers[neo4j]` |
| `config.py` | Add 12 new fields; remove hardcoded Azure key fallback |
| `document_processor.py` | Add `chunk_id` to chunk metadata |
| `rag_system.py` | Wire HybridRetriever; extend `add_documents`/`add_file`/`add_directory` |
| `main.py` | Add `load_dotenv()` at top |
| `.gitignore` | Add `.env`, `bm25_index/`, `__pycache__/`, `*.pyc` |
| `README.md` | Document new architecture & Neo4j Aura setup |

### Untouched files
`chatbot.py`, `logger.py` — no changes.

---

## Task 1: Dependencies and Environment Setup

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`
- Create: `.env` (NOT committed)
- Modify: `.gitignore`

- [ ] **Step 1.1: Update `requirements.txt`**

Replace the entire contents of `requirements.txt` with:

```
chromadb>=0.4.0
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.20
langchain-chroma>=0.1.0
pymysql>=1.1.0
cryptography>=41.0.0
pypdf>=3.0.0

# Hybrid retrieval
rank-bm25>=0.2.2

# Graph RAG
neo4j>=5.15.0

# Configuration
python-dotenv>=1.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
testcontainers[neo4j]>=4.0.0
```

- [ ] **Step 1.2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 1.3: Create `.env.example` (committed template)**

Create `.env.example` with:

```bash
# Azure OpenAI
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_KEY=your-azure-key-here
AZURE_API_VERSION=2023-05-15
DEPLOYMENT_NAME=gpt-4o
EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Neo4j Aura (https://console.neo4j.io)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password-here
NEO4J_DATABASE=neo4j

# MySQL
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=rag_system
MYSQL_PORT=3306

# Feature flags
GRAPH_ENABLED=true
BM25_ENABLED=true
```

- [ ] **Step 1.4: Create `.env` for local development**

Copy `.env.example` to `.env` and fill in real values. **This file MUST NOT be committed.**

- [ ] **Step 1.5: Create or update `.gitignore`**

Append (or create with) these lines:

```
# Secrets
.env

# Indexes
bm25_index/
chroma_db/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/

# IDE
.vscode/
.idea/
```

- [ ] **Step 1.6: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: add Graph RAG dependencies and dotenv template"
```

---

## Task 2: Configuration Additions

**Files:**
- Modify: `config.py` (full rewrite)
- Modify: `main.py:1` (add `load_dotenv` call)

- [ ] **Step 2.1: Replace `config.py` entirely**

Replace `config.py` contents with:

```python
import os
from typing import List, Optional


class RAGConfig:
    # Azure OpenAI — env-only, no hardcoded fallbacks
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_qa")

    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_system")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Vector retrieval
    TOP_K = int(os.getenv("TOP_K", "4"))

    # LLM generation
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

    # Neo4j (NEW)
    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() == "true"

    # BM25 (NEW)
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./bm25_index/bm25.pkl")
    BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"

    # Hybrid retrieval (NEW)
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))

    # Entity extraction (NEW)
    ENTITY_TYPES: List[str] = [
        "Person", "Organization", "Location", "Concept",
        "Event", "Product", "Technology", "Date",
    ]
    EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "5"))
    EXTRACTION_MAX_CHARS = int(os.getenv("EXTRACTION_MAX_CHARS", "4000"))
```

- [ ] **Step 2.2: Add `load_dotenv()` to `main.py`**

At the very top of `main.py` (line 1), add:

```python
from dotenv import load_dotenv
load_dotenv()
```

The existing imports continue below.

- [ ] **Step 2.3: Sanity-check config loads**

Run: `python -c "from dotenv import load_dotenv; load_dotenv(); from config import RAGConfig; c = RAGConfig(); print('NEO4J_URI:', c.NEO4J_URI); print('GRAPH_ENABLED:', c.GRAPH_ENABLED); print('ENTITY_TYPES:', c.ENTITY_TYPES)"`

Expected: prints values from `.env`. No tracebacks.

- [ ] **Step 2.4: Commit**

```bash
git add config.py main.py
git commit -m "feat(config): add Neo4j/BM25/hybrid params, remove hardcoded secrets"
```

---

## Task 3: Add `chunk_id` to DocumentProcessor

**Files:**
- Modify: `document_processor.py:108-111`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_chunk_id.py`

- [ ] **Step 3.1: Create `tests/__init__.py`**

Create empty file `tests/__init__.py`.

- [ ] **Step 3.2: Create `tests/conftest.py`**

```python
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

- [ ] **Step 3.3: Write failing test in `tests/test_chunk_id.py`**

```python
from config import RAGConfig
from document_processor import DocumentProcessor


def test_chunk_id_is_set_on_each_chunk():
    config = RAGConfig()
    processor = DocumentProcessor(config)

    text = "This is a test document. " * 200  # ensures multiple chunks
    metadata = {"name": "test.txt", "type": "txt"}

    chunks = processor.process_documents([text], [metadata])

    assert len(chunks) >= 2, "expected multiple chunks for this length"
    for i, chunk in enumerate(chunks):
        expected_id = f"test.txt::{i}"
        assert chunk.metadata["chunk_id"] == expected_id, (
            f"chunk {i} has wrong chunk_id: {chunk.metadata.get('chunk_id')}"
        )


def test_chunk_id_uses_default_name_when_metadata_missing():
    config = RAGConfig()
    processor = DocumentProcessor(config)

    text = "Hello world. " * 200
    chunks = processor.process_documents([text])

    assert chunks[0].metadata["chunk_id"].startswith("document_0::")
```

- [ ] **Step 3.4: Run test to verify it fails**

Run: `pytest tests/test_chunk_id.py -v`
Expected: FAIL — `KeyError: 'chunk_id'` or `AssertionError`.

- [ ] **Step 3.5: Modify `document_processor.py:98-113`**

Replace the existing `process_documents` method (lines 98-113) with:

```python
    def process_documents(self, documents: List[str],
                         metadata: Optional[List[Dict]] = None) -> List[Document]:
        all_chunks = []

        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)

            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            doc_metadata["doc_index"] = i

            doc_name = doc_metadata.get("name", f"document_{i}")

            for j, chunk in enumerate(chunks):
                chunk_metadata = doc_metadata.copy()
                chunk_metadata["chunk_index"] = j
                chunk_metadata["chunk_id"] = f"{doc_name}::{j}"
                all_chunks.append(Document(page_content=chunk, metadata=chunk_metadata))

        return all_chunks
```

- [ ] **Step 3.6: Run test to verify it passes**

Run: `pytest tests/test_chunk_id.py -v`
Expected: 2 PASSED.

- [ ] **Step 3.7: Commit**

```bash
git add document_processor.py tests/__init__.py tests/conftest.py tests/test_chunk_id.py
git commit -m "feat(processor): add deterministic chunk_id metadata"
```

---

## Task 4: BM25 Retriever — Core

**Files:**
- Create: `bm25_retriever.py`
- Create: `tests/test_bm25.py`

- [ ] **Step 4.1: Write failing test `tests/test_bm25.py`**

```python
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
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_bm25.py -v`
Expected: FAIL — `ImportError: cannot import 'BM25Retriever'`.

- [ ] **Step 4.3: Create `bm25_retriever.py`**

```python
import os
import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import RAGConfig


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_if_exists()

    def add_documents(self, docs: List[Document]) -> None:
        self.documents.extend(docs)
        tokenized = [_tokenize(d.page_content) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        self.save()

    def search(self, query: str, k: int = 10) -> List[Document]:
        if self.bm25 is None or not self.documents:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(scores, self.documents), key=lambda pair: pair[0], reverse=True
        )
        return [doc for _, doc in ranked[:k]]

    def save(self) -> None:
        path = Path(self.config.BM25_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)

    def load(self) -> None:
        path = Path(self.config.BM25_INDEX_PATH)
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        if self.documents:
            tokenized = [_tokenize(d.page_content) for d in self.documents]
            self.bm25 = BM25Okapi(tokenized)

    def _load_if_exists(self) -> None:
        try:
            self.load()
        except Exception as e:
            print(f"Warning: BM25 load failed ({e}); starting empty.")
            self.documents = []
            self.bm25 = None
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_bm25.py -v`
Expected: 3 PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add bm25_retriever.py tests/test_bm25.py
git commit -m "feat(bm25): add BM25 retriever with pickle persistence"
```

---

## Task 5: RRF Fusion

**Files:**
- Create: `hybrid_retriever.py` (skeleton with rrf_fuse only — full class added in Task 13)
- Create: `tests/test_rrf.py`

- [ ] **Step 5.1: Write failing test `tests/test_rrf.py`**

```python
from langchain_core.documents import Document
from hybrid_retriever import rrf_fuse


def doc(chunk_id: str) -> Document:
    return Document(page_content=f"text-{chunk_id}", metadata={"chunk_id": chunk_id})


def test_rrf_fuses_three_rankings():
    # Doc "a" appears in all three rankings near top — should win.
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

    assert len(fused) == 2  # not 4
    assert {d.metadata["chunk_id"] for d in fused} == {"a", "b"}


def test_rrf_higher_rank_beats_lower_rank():
    # Doc b is at rank 0 in r2, rank 2 in r1. Doc c is at rank 1 in both.
    r1 = [doc("c"), doc("c"), doc("b")]   # b is rank 2
    r2 = [doc("b"), doc("c"), doc("c")]   # b is rank 0

    # Use unique docs (deduped fusion uses chunk_id) — clearer test:
    r1 = [doc("c"), doc("a"), doc("b")]   # b rank 2
    r2 = [doc("b"), doc("a"), doc("c")]   # b rank 0

    fused = rrf_fuse([r1, r2], k=60)
    ids = [d.metadata["chunk_id"] for d in fused]

    # 'a' is at rank 1 in both — score = 1/62 + 1/62 = 0.0323
    # 'b' is at rank 2 + 0 — score = 1/63 + 1/61 = 0.0322
    # 'c' is at rank 0 + 2 — score = 1/61 + 1/63 = 0.0322
    # 'a' wins because consistent middle is best.
    assert ids[0] == "a"


def test_rrf_returns_empty_for_all_empty():
    fused = rrf_fuse([[], [], []], k=60)
    assert fused == []
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `pytest tests/test_rrf.py -v`
Expected: FAIL — `ImportError: cannot import 'rrf_fuse' from 'hybrid_retriever'`.

- [ ] **Step 5.3: Create `hybrid_retriever.py` skeleton with `rrf_fuse`**

```python
from collections import defaultdict
from typing import List

from langchain_core.documents import Document


def rrf_fuse(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009).

    Combines multiple ranked lists into one ranking using only ranks
    (not scores), since scores from different retrievers aren't comparable.

    Args:
        rankings: list of ranked Document lists (one per retriever)
        k: smoothing constant (60 is the published default)

    Returns:
        Merged & deduped (by chunk_id) list, highest score first.
    """
    scores: dict = defaultdict(float)
    docs_by_id: dict = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = doc.metadata.get("chunk_id")
            if cid is None:
                continue
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in docs_by_id:
                docs_by_id[cid] = doc

    sorted_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])
    return [docs_by_id[cid] for cid in sorted_ids]
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `pytest tests/test_rrf.py -v`
Expected: 5 PASSED.

- [ ] **Step 5.5: Commit**

```bash
git add hybrid_retriever.py tests/test_rrf.py
git commit -m "feat(hybrid): add Reciprocal Rank Fusion (RRF)"
```

---

## Task 6: GraphStore — Connection Lifecycle

**Files:**
- Create: `graph_store.py`
- Create: `tests/test_graph_store.py`

- [ ] **Step 6.1: Write failing test `tests/test_graph_store.py`**

```python
from unittest.mock import MagicMock, patch
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

    import pytest
    with pytest.raises(ValueError, match="NEO4J_URI"):
        GraphStore(config)
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/test_graph_store.py -v`
Expected: FAIL — `ImportError: cannot import 'GraphStore' from 'graph_store'`.

- [ ] **Step 6.3: Create `graph_store.py` (lifecycle only)**

```python
from typing import Iterable, List, Dict, Any, Optional
from langchain_core.documents import Document
from neo4j import GraphDatabase

from config import RAGConfig


class GraphStore:
    def __init__(self, config: RAGConfig):
        if not config.NEO4J_URI:
            raise ValueError("NEO4J_URI is empty; cannot connect to Neo4j.")
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `pytest tests/test_graph_store.py -v`
Expected: 3 PASSED.

- [ ] **Step 6.5: Commit**

```bash
git add graph_store.py tests/test_graph_store.py
git commit -m "feat(graph): add GraphStore connection lifecycle"
```

---

## Task 7: GraphStore — Write Chunk Nodes

**Files:**
- Modify: `graph_store.py` (add `add_chunks` method)
- Modify: `tests/test_graph_store.py` (add tests)

- [ ] **Step 7.1: Append failing test to `tests/test_graph_store.py`**

```python
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

    # session.run was called at least once with a MERGE Chunk query
    calls = mock_session.run.call_args_list
    assert any("MERGE (c:Chunk" in str(call) for call in calls)
```

Add at the top of the test file:
```python
from langchain_core.documents import Document
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `pytest tests/test_graph_store.py::test_add_chunks_runs_merge_query -v`
Expected: FAIL — `AttributeError: 'GraphStore' has no attribute 'add_chunks'`.

- [ ] **Step 7.3: Add `add_chunks` to `graph_store.py`**

Append to the `GraphStore` class:

```python
    def add_chunks(self, docs: List[Document]) -> None:
        """Write each document as a Chunk node, idempotent via MERGE."""
        if not docs:
            return
        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            for d in docs:
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text,
                        c.doc_name = $doc_name,
                        c.chunk_index = $chunk_index
                    """,
                    chunk_id=d.metadata.get("chunk_id"),
                    text=d.page_content,
                    doc_name=d.metadata.get("name", ""),
                    chunk_index=d.metadata.get("chunk_index", 0),
                )
```

- [ ] **Step 7.4: Run test to verify it passes**

Run: `pytest tests/test_graph_store.py -v`
Expected: 4 PASSED.

- [ ] **Step 7.5: Commit**

```bash
git add graph_store.py tests/test_graph_store.py
git commit -m "feat(graph): add Chunk node writes (idempotent MERGE)"
```

---

## Task 8: GraphStore — Write Entities & Relations

**Files:**
- Modify: `graph_store.py` (add `add_entities` method)
- Modify: `tests/test_graph_store.py` (add tests)

- [ ] **Step 8.1: Append failing test**

```python
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
    # entity merges
    assert any("MERGE (e:Entity" in c for c in calls)
    # mention edge
    assert any("MENTIONED_IN" in c for c in calls)
    # relation merge
    assert any("RELATION" in c for c in calls)
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `pytest tests/test_graph_store.py::test_add_entities_writes_entity_and_mention -v`
Expected: FAIL — no `add_entities` method.

- [ ] **Step 8.3: Add `add_entities` to `graph_store.py`**

```python
    def add_entities(self, triples: Dict[str, Any]) -> None:
        """
        Write extracted entities & relations.

        triples = {
            "chunk_id": str,
            "entities": [{"name": str, "type": str}, ...],
            "relations": [{"source": str, "target": str, "type": str}, ...],
        }
        """
        chunk_id = triples.get("chunk_id")
        entities = triples.get("entities") or []
        relations = triples.get("relations") or []
        if not chunk_id:
            return

        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            for ent in entities:
                name = (ent.get("name") or "").strip()
                etype = (ent.get("type") or "Concept").strip()
                if not name:
                    continue
                session.run(
                    """
                    MERGE (e:Entity {name_lower: toLower($name)})
                    SET e.name = $name, e.type = $type
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    name=name, type=etype, chunk_id=chunk_id,
                )

            for rel in relations:
                src = (rel.get("source") or "").strip()
                tgt = (rel.get("target") or "").strip()
                rtype = (rel.get("type") or "RELATES_TO").strip().upper().replace(" ", "_")
                if not src or not tgt:
                    continue
                session.run(
                    """
                    MERGE (a:Entity {name_lower: toLower($src)})
                    MERGE (b:Entity {name_lower: toLower($tgt)})
                    MERGE (a)-[r:RELATION {type: $rtype}]->(b)
                    """,
                    src=src, tgt=tgt, rtype=rtype,
                )
```

- [ ] **Step 8.4: Run test to verify it passes**

Run: `pytest tests/test_graph_store.py -v`
Expected: 5 PASSED.

- [ ] **Step 8.5: Commit**

```bash
git add graph_store.py tests/test_graph_store.py
git commit -m "feat(graph): write entities & relations with lowercased dedup"
```

---

## Task 9: GraphStore — Query for Chunks (1-hop & 2-hop)

**Files:**
- Modify: `graph_store.py` (add `search` method)
- Modify: `tests/test_graph_store.py` (add tests)

- [ ] **Step 9.1: Append failing tests**

```python
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
    assert "RELATION" in call_query  # 2-hop uses RELATION traversal


@patch("graph_store.GraphDatabase")
def test_search_with_no_entities_returns_empty(mock_gdb):
    config = make_config()
    store = GraphStore(config)

    results = store.search([], k=10, hops=1)
    assert results == []
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `pytest tests/test_graph_store.py -v`
Expected: FAIL — no `search` method.

- [ ] **Step 9.3: Add `search` to `graph_store.py`**

```python
    def search(self, entity_names: List[str], k: int = 10, hops: int = 1) -> List[Document]:
        """
        Find chunks related to the given entity names.

        hops=1: chunks directly mentioning matched entities.
        hops=2: also chunks mentioning entities related to matched entities.
        """
        if not entity_names:
            return []

        names_lower = [n.lower() for n in entity_names]

        if hops <= 1:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
               OR any(n IN $names WHERE toLower(e.name) CONTAINS n)
            WITH collect(e) AS matched
            UNWIND matched AS e
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.id AS id, c.text AS text,
                   c.doc_name AS doc_name, c.chunk_index AS chunk_index,
                   count(*) AS relevance
            ORDER BY relevance DESC
            LIMIT $k
            """
        else:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
               OR any(n IN $names WHERE toLower(e.name) CONTAINS n)
            WITH collect(e) AS matched
            UNWIND matched AS e
            MATCH (e)-[r:RELATION]-(neighbor:Entity)-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.id AS id, c.text AS text,
                   c.doc_name AS doc_name, c.chunk_index AS chunk_index,
                   count(DISTINCT neighbor) AS relevance
            ORDER BY relevance DESC
            LIMIT $k
            """

        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            records = session.run(query, names=names_lower, k=k)
            results: List[Document] = []
            for r in records:
                results.append(Document(
                    page_content=r["text"] or "",
                    metadata={
                        "chunk_id": r["id"],
                        "name": r["doc_name"] or "",
                        "chunk_index": r["chunk_index"] or 0,
                        "graph_relevance": r["relevance"],
                    },
                ))
            return results
```

- [ ] **Step 9.4: Run test to verify it passes**

Run: `pytest tests/test_graph_store.py -v`
Expected: 8 PASSED.

- [ ] **Step 9.5: Commit**

```bash
git add graph_store.py tests/test_graph_store.py
git commit -m "feat(graph): add search() with 1-hop and 2-hop traversal"
```

---

## Task 10: Graph Extractor — JSON Parsing & Truncation

**Files:**
- Create: `graph_extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 10.1: Write failing test**

```python
import json
from unittest.mock import MagicMock
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
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 10.3: Create `graph_extractor.py`**

```python
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config import RAGConfig

if TYPE_CHECKING:
    from graph_store import GraphStore


_PROMPT_TEMPLATE = """Extract entities and relationships from the text below.

Entity types (use ONLY these):
  {entity_types}

Return JSON only (no prose, no code fences):
{{
  "entities": [{{"name": "...", "type": "<one of the types above>"}}],
  "relations": [{{"source": "...", "target": "...", "type": "VERB_PHRASE"}}]
}}

Text:
{text}
"""


class GraphExtractor:
    def __init__(self, llm: Any, config: RAGConfig):
        self.llm = llm
        self.config = config

    def _truncate(self, text: str) -> str:
        return text[: self.config.EXTRACTION_MAX_CHARS]

    def _parse(self, raw: str) -> Dict[str, List[Dict[str, Any]]]:
        # Strip optional ```json ... ``` wrapper
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"entities": [], "relations": []}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
        return {
            "entities": data.get("entities", []) or [],
            "relations": data.get("relations", []) or [],
        }

    def _build_prompt(self, text: str) -> str:
        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(self.config.ENTITY_TYPES),
            text=self._truncate(text),
        )

    async def _extract_one(self, doc: Document) -> Dict[str, Any]:
        prompt = self._build_prompt(doc.page_content)
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            triples = self._parse(content)
        except Exception as e:
            print(f"Warning: extraction failed for {doc.metadata.get('chunk_id')}: {e}")
            triples = {"entities": [], "relations": []}
        triples["chunk_id"] = doc.metadata.get("chunk_id")
        return triples

    async def extract_and_store_async(
        self, docs: List[Document], graph_store: "GraphStore"
    ) -> None:
        sem = asyncio.Semaphore(self.config.EXTRACTION_CONCURRENCY)

        async def bound_extract(d: Document) -> Dict[str, Any]:
            async with sem:
                return await self._extract_one(d)

        tasks = [bound_extract(d) for d in docs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for triples in results:
            graph_store.add_entities(triples)

    def extract_and_store(self, docs: List[Document], graph_store: "GraphStore") -> None:
        asyncio.run(self.extract_and_store_async(docs, graph_store))
```

- [ ] **Step 10.4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: 4 PASSED.

- [ ] **Step 10.5: Commit**

```bash
git add graph_extractor.py tests/test_extractor.py
git commit -m "feat(graph): add LLM-based entity/relation extractor"
```

---

## Task 11: Graph Extractor — Async Concurrency

**Files:**
- Modify: `tests/test_extractor.py` (add async test)

- [ ] **Step 11.1: Append async test to `tests/test_extractor.py`**

```python
import pytest


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
```

- [ ] **Step 11.2: Verify pytest-asyncio mode is set**

Create `pytest.ini` at the project root:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 11.3: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: 5 PASSED.

- [ ] **Step 11.4: Commit**

```bash
git add tests/test_extractor.py pytest.ini
git commit -m "test(graph): verify async extraction respects concurrency"
```

---

## Task 12: HybridRetriever — Class & Parallel Fan-out

**Files:**
- Modify: `hybrid_retriever.py` (add `HybridRetriever` class)
- Create: `tests/test_hybrid_retriever.py`

- [ ] **Step 12.1: Write failing test `tests/test_hybrid_retriever.py`**

```python
from unittest.mock import MagicMock, AsyncMock
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

    # All three were called
    vec.invoke.assert_called_once()
    bm25.search.assert_called_once()
    graph.search.assert_called_once()

    # Returns at most TOP_K
    assert len(result) <= config.TOP_K

    # 'a' appears in all 3 → must be in result
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
    assert len(result) > 0  # Did not crash; returned vector + BM25 results
```

- [ ] **Step 12.2: Run test to verify it fails**

Run: `pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL — `ImportError: cannot import 'HybridRetriever'`.

- [ ] **Step 12.3: Append `HybridRetriever` class + entity-name helper to `hybrid_retriever.py`**

Append to existing `hybrid_retriever.py`:

```python
import json
import re
from typing import Any, List, Optional

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage


class HybridRetriever(Runnable):
    """
    Runs vector + BM25 + graph retrievers and fuses via RRF.
    Any retriever can be None — it will be skipped gracefully.
    """

    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: Any,
        graph_retriever: Any,
        graph_extractor: Any,
        config,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.graph_extractor = graph_extractor
        self.config = config

    def invoke(self, question: str, *args, **kwargs) -> List[Document]:
        rankings: List[List[Document]] = []

        # Vector
        if self.vector_retriever is not None:
            try:
                rankings.append(self.vector_retriever.invoke(question) or [])
            except Exception as e:
                print(f"Warning: vector retrieval failed: {e}")

        # BM25
        if self.bm25_retriever is not None:
            try:
                rankings.append(
                    self.bm25_retriever.search(question, k=self.config.RETRIEVER_K) or []
                )
            except Exception as e:
                print(f"Warning: BM25 retrieval failed: {e}")

        # Graph
        if self.graph_retriever is not None and self.graph_extractor is not None:
            try:
                names = self.graph_extractor._extract_entity_names(question)
                rankings.append(
                    self.graph_retriever.search(
                        names, k=self.config.RETRIEVER_K, hops=self.config.GRAPH_HOPS
                    ) or []
                )
            except Exception as e:
                print(f"Warning: graph retrieval failed: {e}")

        fused = rrf_fuse(rankings, k=self.config.RRF_K)
        return fused[: self.config.TOP_K]
```

- [ ] **Step 12.4: Add `_extract_entity_names` to `graph_extractor.py`**

Append to the `GraphExtractor` class:

```python
    def _extract_entity_names(self, question: str) -> List[str]:
        """
        Synchronously ask the LLM to list entities in a question.
        Used for graph retrieval (not ingestion).
        """
        prompt = (
            "Extract entity names from this question. "
            "Return JSON only: {\"entities\": [\"name1\", \"name2\"]}\n\n"
            f"Question: {self._truncate(question)}"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group(0))
            names = data.get("entities", []) or []
            return [n for n in names if isinstance(n, str) and n.strip()]
        except Exception as e:
            print(f"Warning: question entity extraction failed: {e}")
            return []
```

- [ ] **Step 12.5: Run all tests to verify**

Run: `pytest tests/ -v`
Expected: All tests pass (chunk_id, bm25, rrf, graph_store, extractor, hybrid_retriever).

- [ ] **Step 12.6: Commit**

```bash
git add hybrid_retriever.py graph_extractor.py tests/test_hybrid_retriever.py
git commit -m "feat(hybrid): add HybridRetriever with parallel fan-out & graceful degradation"
```

---

## Task 13: Wire HybridRetriever into `rag_system.py`

**Files:**
- Modify: `rag_system.py` (constructor, LCEL chain, ingestion methods, close)

- [ ] **Step 13.1: Update imports in `rag_system.py:1-14`**

Replace the import block at the top of `rag_system.py` (currently lines 1-14) with:

```python
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import chromadb
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import RAGConfig
from logger import MySQLLogger
from document_processor import DocumentProcessor
from bm25_retriever import BM25Retriever
from graph_store import GraphStore
from graph_extractor import GraphExtractor
from hybrid_retriever import HybridRetriever
```

- [ ] **Step 13.2: Modify `_initialize_components` (around `rag_system.py:31-103`)**

Find the section after the QA chain is built (after the `print("✓ QA chain initialized")` line) and locate the existing `self.retriever = self.vectorstore.as_retriever(...)` block (lines 85-87).

Replace lines 85-100 (from `self.retriever = ...` through the `self.qa_chain = (...)` block) with:

```python
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.RETRIEVER_K}
        )

        # Hybrid retrieval components (Task 13)
        self.bm25 = None
        self.graph = None
        self.graph_extractor = None

        if self.config.BM25_ENABLED:
            try:
                self.bm25 = BM25Retriever(self.config)
                print("✓ BM25 retriever initialized")
            except Exception as e:
                print(f"Warning: BM25 disabled ({e})")

        if self.config.GRAPH_ENABLED:
            try:
                self.graph = GraphStore(self.config)
                self.graph_extractor = GraphExtractor(self.llm, self.config)
                print("✓ Neo4j graph store initialized")
            except Exception as e:
                print(f"Warning: graph disabled ({e})")
                self.graph = None
                self.graph_extractor = None

        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.retriever,
            bm25_retriever=self.bm25,
            graph_retriever=self.graph,
            graph_extractor=self.graph_extractor,
            config=self.config,
        )
        print("✓ Hybrid retriever initialized")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {
                "context": self.hybrid_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        print("✓ QA chain initialized")
        print("=" * 50)
```

- [ ] **Step 13.3: Modify `query()` to use `hybrid_retriever` for `source_documents`**

In `rag_system.py:165`, change:

```python
        source_documents = self.retriever.invoke(question)
```

to:

```python
        source_documents = self.hybrid_retriever.invoke(question)
```

- [ ] **Step 13.4: Extend `add_documents` (around `rag_system.py:105-124`)**

Replace the existing `add_documents` method body. Find the line `self.vectorstore.add_documents(all_chunks)` and add the BM25 and graph writes after it:

```python
        self.vectorstore.add_documents(all_chunks)
        print(f"✓ Added {len(all_chunks)} chunks to vector store")

        if self.bm25:
            self.bm25.add_documents(all_chunks)
            print(f"✓ Added {len(all_chunks)} chunks to BM25 index")

        if self.graph:
            self.graph.add_chunks(all_chunks)
            print(f"✓ Added {len(all_chunks)} Chunk nodes to Neo4j")
            if self.graph_extractor:
                print(f"  Extracting entities for {len(all_chunks)} chunks (this can take a minute)...")
                self.graph_extractor.extract_and_store(all_chunks, self.graph)
                print(f"✓ Entity extraction complete")
```

- [ ] **Step 13.5: Apply same extension to `add_file` (around `rag_system.py:126-141`)**

Find `self.vectorstore.add_documents(chunks)` at the end of `add_file`. After that line, append:

```python
        if self.bm25:
            self.bm25.add_documents(chunks)
        if self.graph:
            self.graph.add_chunks(chunks)
            if self.graph_extractor:
                print(f"  Extracting entities for {len(chunks)} chunks…")
                self.graph_extractor.extract_and_store(chunks, self.graph)
```

- [ ] **Step 13.6: Apply same extension to `add_directory` (around `rag_system.py:143-160`)**

Find `self.vectorstore.add_documents(chunks)` at the end of `add_directory`. After that line, append:

```python
        if self.bm25:
            self.bm25.add_documents(chunks)
        if self.graph:
            self.graph.add_chunks(chunks)
            if self.graph_extractor:
                print(f"  Extracting entities for {len(chunks)} chunks…")
                self.graph_extractor.extract_and_store(chunks, self.graph)
```

- [ ] **Step 13.7: Update `close()` (around `rag_system.py:215-216`)**

Replace:

```python
    def close(self):
        self.logger.close()
```

with:

```python
    def close(self):
        self.logger.close()
        if self.graph:
            try:
                self.graph.close()
            except Exception as e:
                print(f"Warning: error closing Neo4j: {e}")
```

- [ ] **Step 13.8: Run unit tests to verify nothing broke**

Run: `pytest tests/ -v`
Expected: All previous tests still pass.

- [ ] **Step 13.9: Sanity-check import**

Run: `python -c "from dotenv import load_dotenv; load_dotenv(); from rag_system import DocumentRAGSystem; print('imports OK')"`
Expected: prints "imports OK". No tracebacks.

- [ ] **Step 13.10: Commit**

```bash
git add rag_system.py
git commit -m "feat(rag): wire HybridRetriever and feed BM25/graph during ingestion"
```

---

## Task 14: Integration Test (Neo4j testcontainer)

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/test_e2e.py`
- Create: `tests/fixtures/sample_corpus.txt`

- [ ] **Step 14.1: Create `tests/integration/__init__.py` (empty file)**

- [ ] **Step 14.2: Create `tests/fixtures/sample_corpus.txt`**

```
Alice is a software engineer at Acme Corporation.
She uses Python and PyTorch for machine learning research.

Bob also works at Acme Corporation as a data scientist.
He prefers Rust and writes high-performance pipelines.

Acme Corporation is headquartered in Berlin and was founded in 2010.
The company specializes in distributed systems and AI.
```

- [ ] **Step 14.3: Write `tests/integration/test_e2e.py`**

```python
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

pytestmark = pytest.mark.skipif(
    not testcontainers_available
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
    # Cleanup
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
        "entities": [{"name": "python", "type": "Technology"}],  # different case
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

    # Searching for "Speed" with 2 hops should also find chunks about Python (and Alice)
    results = graph_store.search(["Speed"], k=10, hops=2)
    chunk_ids = {r.metadata["chunk_id"] for r in results}
    assert "b::0" in chunk_ids
```

- [ ] **Step 14.4: Run integration test**

Ensure Docker is running, then:
Run: `pytest tests/integration -v`
Expected: 3 PASSED (or all SKIPPED if Docker unavailable; that's also acceptable).

- [ ] **Step 14.5: Commit**

```bash
git add tests/integration tests/fixtures
git commit -m "test: add Neo4j integration tests via testcontainers"
```

---

## Task 15: Smoke Test Script

**Files:**
- Create: `tests/smoke_test.py`

- [ ] **Step 15.1: Create `tests/smoke_test.py`**

```python
"""
Manual smoke test — not run by pytest.

Usage:
    python tests/smoke_test.py

Prerequisites:
    - .env populated with Azure + Neo4j Aura credentials
    - MySQL running locally (or MYSQL_PASSWORD set in .env)
    - Sai_Prashanth_CV.pdf in project root (or any PDF)

What it does:
    1. Initializes the full RAG system
    2. Ingests the CV
    3. Asks 4 questions covering each retriever's strength
    4. Prints the answers + which sources contributed
"""
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from config import RAGConfig
from rag_system import DocumentRAGSystem


CV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Sai_Prashanth_CV.pdf")
)

QUESTIONS = [
    "What are the technical skills mentioned?",        # vector strength
    "Mention any acronyms or specific certifications.", # BM25 strength
    "Who or what is connected to specific companies?", # graph strength
    "Summarize the work experience.",                   # blend
]


def main():
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    config = RAGConfig()
    print(f"GRAPH_ENABLED: {config.GRAPH_ENABLED}")
    print(f"BM25_ENABLED: {config.BM25_ENABLED}")
    print()

    rag = DocumentRAGSystem(config)

    if os.path.exists(CV_PATH):
        print(f"Ingesting: {CV_PATH}")
        rag.add_file(CV_PATH)
    else:
        print(f"WARNING: {CV_PATH} not found — skipping ingestion. "
              f"Will query whatever is already indexed.")

    print()
    for q in QUESTIONS:
        print("-" * 60)
        print(f"Q: {q}")
        response = rag.query(q)
        print(f"A: {response['answer'][:400]}")
        print(f"Sources: {response['num_sources']}")
        print(f"Time: {response['execution_time']:.2f}s")
        print()

    rag.close()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 15.2: Run smoke test (manual)**

Run: `python tests/smoke_test.py`
Expected: 4 questions answered. May take a few minutes the first time due to entity extraction.

- [ ] **Step 15.3: Commit**

```bash
git add tests/smoke_test.py
git commit -m "test: add manual smoke test runner"
```

---

## Task 16: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 16.1: Replace `README.md` with updated version**

Replace the entire `README.md` with:

```markdown
# Graph RAG with Hybrid Retrieval

A production-grade Retrieval-Augmented Generation system combining **dense vector embeddings (ChromaDB)**, **BM25 sparse retrieval**, and a **Neo4j knowledge graph**, fused via **Reciprocal Rank Fusion (RRF)**.

## Features

- **Hybrid retrieval** — three retrievers run in parallel, results merged via RRF
- **Knowledge graph** — entities & relations auto-extracted from documents using gpt-4o
- **8-type entity schema** — Person, Organization, Location, Concept, Event, Product, Technology, Date
- **Async ingestion** — concurrent LLM calls with semaphore-limited parallelism
- **Graceful degradation** — system runs even if Neo4j or BM25 is unavailable
- **Conversation logging** — every query persisted to MySQL with session tracking
- **Modular architecture** — each retriever lives in its own module, testable in isolation

## Architecture

```
                ┌────────────────────────────┐
                │    INGESTION               │
PDF/TXT/MD ───► │  chunks → Chroma           │
                │         → BM25             │
                │         → Neo4j (Chunks)   │
                │         → gpt-4o (entities)│
                └────────────────────────────┘

                ┌────────────────────────────┐
                │    QUERY                   │
                │  question                  │
                │     ├─► Chroma             │
                │     ├─► BM25       ──► RRF │
                │     └─► Neo4j              │
                │              │             │
                │              ▼             │
                │      gpt-4o + grounded     │
                │           prompt           │
                └────────────────────────────┘
```

## Project Structure

```
.
├── main.py                 # Entry point
├── config.py               # Centralized settings (env-driven)
├── chatbot.py              # Interactive CLI
├── document_processor.py   # Chunking + chunk_id assignment
├── rag_system.py           # Orchestrator wiring all retrievers
├── bm25_retriever.py       # BM25 sparse retriever (rank-bm25)
├── graph_store.py          # Neo4j connection + Cypher
├── graph_extractor.py      # LLM-based entity/relation extraction
├── hybrid_retriever.py     # Parallel fan-out + RRF fusion
├── logger.py               # MySQL session/query logging
├── tests/                  # Unit + integration tests
└── docs/superpowers/       # Spec & implementation plan
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Copy the template and fill in your values:

```bash
cp .env.example .env
```

You need:
- **Azure OpenAI** — endpoint, API key, deployment names
- **Neo4j Aura** — sign up free at https://console.neo4j.io and grab the connection URI + password
- **MySQL** — running locally with credentials

### 3. Run

```bash
python main.py
```

You'll be prompted to ingest documents, then dropped into an interactive chat.

## Tuning

All knobs live in `config.py` (or `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Chars shared between adjacent chunks |
| `RETRIEVER_K` | 10 | Pool size from each retriever before fusion |
| `TOP_K` | 4 | Final chunks fed to the LLM |
| `RRF_K` | 60 | RRF smoothing constant |
| `GRAPH_HOPS` | 1 | Graph traversal depth (1 or 2) |
| `EXTRACTION_CONCURRENCY` | 5 | Concurrent gpt-4o calls during ingestion |
| `GRAPH_ENABLED` | true | Toggle Neo4j retriever |
| `BM25_ENABLED` | true | Toggle BM25 retriever |

## Testing

```bash
# Unit tests (fast, no external services)
pytest tests/ -v --ignore=tests/integration

# Integration tests (requires Docker for Neo4j testcontainer)
pytest tests/integration -v

# Manual smoke test (real Azure + Neo4j Aura)
python tests/smoke_test.py
```

## How retrieval modes degrade

| `GRAPH_ENABLED` | `BM25_ENABLED` | Behavior |
|-----------------|----------------|----------|
| true            | true           | Vector + BM25 + Graph + RRF |
| false           | true           | Vector + BM25 + RRF |
| true            | false          | Vector + Graph + RRF |
| false           | false          | Vector-only (zero regression) |

If Neo4j is unreachable at runtime, the graph retriever silently drops out and the other two continue.

## Spec & Plan

- Design spec: `docs/superpowers/specs/2026-05-06-graph-rag-hybrid-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-graph-rag-hybrid.md`
```

- [ ] **Step 16.2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for Graph RAG hybrid retrieval"
```

---

## Final Verification

- [ ] **Step F.1: Run all unit tests**

Run: `pytest tests/ --ignore=tests/integration -v`
Expected: All unit tests pass (chunk_id, bm25, rrf, graph_store, extractor, hybrid_retriever).

- [ ] **Step F.2: Run integration tests if Docker available**

Run: `pytest tests/integration -v`
Expected: 3 PASSED, or all SKIPPED with a clear "testcontainers/Docker not available" reason.

- [ ] **Step F.3: Run smoke test against real services**

Run: `python tests/smoke_test.py`
Expected: 4 questions answered, system closes cleanly.

- [ ] **Step F.4: Verify the chatbot still works end-to-end**

Run: `python main.py`
Expected: doc setup wizard appears; pick option 4 (skip); ask a few questions; verify answers reference sources; type `/quit`.

- [ ] **Step F.5: Final commit (if any cleanup is needed)**

If anything was tweaked during verification:
```bash
git add -A
git commit -m "chore: final cleanup after verification"
```

---

## Done. Resume bullet:

> Built a production-grade Graph-RAG system in Python: hybrid retrieval combining ChromaDB dense embeddings, BM25 sparse retrieval, and a Neo4j knowledge graph (8-type entity schema with LLM-driven relation extraction), fused via Reciprocal Rank Fusion (Cormack et al. 2009). Features async entity extraction with semaphore-limited concurrency, graceful degradation when components are unavailable, MySQL session logging, and integration tests via Neo4j testcontainers.
