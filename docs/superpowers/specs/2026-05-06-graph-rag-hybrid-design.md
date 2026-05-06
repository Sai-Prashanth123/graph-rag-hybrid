# Graph RAG with Hybrid Retrieval — Design Spec

**Date:** 2026-05-06
**Status:** Approved (pending user spec review)
**Project:** General-purpose RAG system (`D:\GenAI-Project\genai`)

---

## 1. Goal

Extend the existing vector-only RAG system into a production-grade **Graph RAG with hybrid retrieval**:

- Add a Neo4j knowledge graph extracted from documents during ingestion.
- Add BM25 sparse retrieval alongside the existing dense vector retrieval.
- Fuse all three retrievers (vector + BM25 + graph) using Reciprocal Rank Fusion (RRF).
- Keep the system general-purpose (any document type, not CV-specific).
- Preserve all existing functionality (chatbot CLI, MySQL logging, Chroma persistence).

The system must degrade gracefully: if any retriever or backend is unavailable, the rest still work.

---

## 2. Non-Goals

- LLM-based query routing (we always run all three retrievers and let RRF pick winners).
- A web UI — CLI-only, same as today.
- Replacing MySQL or ChromaDB.
- Re-implementing chunking — the existing `RecursiveCharacterTextSplitter` is unchanged.
- Optimizing extraction cost beyond async parallelism.

---

## 3. Architecture

### 3.1 High-level shape

```
                          ┌─────────────────────────────────┐
                          │   INGESTION (per document)      │
                          │                                 │
   PDF/TXT/MD ───────────►│  1. Chunk (existing)            │
                          │  2. Embed → Chroma (existing)   │
                          │  3. Index → BM25 (NEW)          │
                          │  4. Extract entities/relations  │
                          │     via gpt-4o → Neo4j (NEW)    │
                          └─────────────────────────────────┘

                          ┌─────────────────────────────────┐
                          │   QUERY (per question)          │
                          │                                 │
                          │   Question                      │
                          │      │                          │
                          │      ├──► Chroma  ──┐           │
   User asks ────────────►│      ├──► BM25    ──┤           │
                          │      └──► Neo4j   ──┤           │
                          │                    ▼            │
                          │           RRF Fusion (NEW)      │
                          │                    │            │
                          │                    ▼            │
                          │       Top-K merged chunks       │
                          │                    │            │
                          │                    ▼            │
                          │           gpt-4o + grounded     │
                          │           prompt (existing)     │
                          │                    │            │
                          │                    ▼            │
                          │              Answer             │
                          └─────────────────────────────────┘
```

### 3.2 Module dependency graph

```
              ┌──────────────┐
              │  rag_system  │  (orchestrator, mostly unchanged)
              └──┬───────────┘
                 │
       ┌─────────┼──────────┬──────────────┐
       ▼         ▼          ▼              ▼
   Chroma    BM25       GraphStore   GraphExtractor
              ↑              ↑              │
              │              │              ▼
              └──────────────┴────► HybridRetriever
                                    (RRF fusion)
```

Each new module has a single responsibility, hides its backend, and is testable in isolation.

---

## 4. Components

### 4.1 New files

| File | Lines (approx) | Role |
|------|---------------|------|
| `bm25_retriever.py` | ~80 | Wraps `rank-bm25` library. Tokenizes, indexes, retrieves. Pickled to `./bm25_index/bm25.pkl`. |
| `graph_extractor.py` | ~120 | Async LLM-based entity/relation extraction from chunks. Returns triples to write to Neo4j. |
| `graph_store.py` | ~150 | Neo4j connection management + Cypher queries (write entities, retrieve by entity match + traversal). |
| `hybrid_retriever.py` | ~100 | Runs all 3 retrievers in parallel via `asyncio.gather`, fuses via RRF, returns top-K chunks. |

### 4.2 Modified files

| File | Change |
|------|--------|
| `config.py` | Add Neo4j credentials, BM25 + RRF + extraction tuning params, feature flags. Remove hardcoded Azure key. |
| `rag_system.py` | Conditionally instantiate BM25 + Graph + Extractor + HybridRetriever; wire HybridRetriever into LCEL chain; extend `add_documents()`/`add_file()`/`add_directory()` to feed BM25 + graph. |
| `requirements.txt` | Add `neo4j`, `rank-bm25`, `python-dotenv`. |
| `main.py` | Add `from dotenv import load_dotenv; load_dotenv()` at top. |
| `.gitignore` | Add `.env`, `bm25_index/`. |

### 4.3 New non-code files

| File | Purpose |
|------|---------|
| `.env.example` | Template with placeholder keys. Committed. |
| `.env` | Real credentials. **Gitignored.** |
| `tests/test_hybrid_rrf.py` | Pure unit tests for RRF math. |
| `tests/test_bm25.py` | Unit tests for BM25 retriever. |
| `tests/test_extractor.py` | Unit tests for entity-extraction JSON parsing (mocks LLM). |
| `tests/test_graph_store.py` | Unit tests for Cypher query construction. |
| `tests/integration/test_e2e.py` | Integration tests using `testcontainers-python` (Neo4j). |
| `tests/fixtures/sample_corpus.txt` | 3 small bios with known entity overlap, for assertions. |
| `tests/smoke_test.py` | Manual end-to-end smoke runner against the real CV. |

### 4.4 Untouched files

- `chatbot.py` — interactive CLI is unchanged.
- `document_processor.py` — chunking is unchanged.
- `logger.py` — MySQL logging is unchanged.

---

## 5. Configuration (`config.py`)

New fields added to `RAGConfig`:

```python
# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() == "true"

# BM25
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./bm25_index/bm25.pkl")
BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"

# Hybrid retrieval
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))   # per-retriever pool size before fusion
RRF_K = int(os.getenv("RRF_K", "60"))               # RRF constant from Cormack et al. 2009
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))      # 1 or 2 hop traversal

# Entity extraction
ENTITY_TYPES = ["Person", "Organization", "Location", "Concept",
                "Event", "Product", "Technology", "Date"]
EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "5"))
EXTRACTION_MAX_CHARS = int(os.getenv("EXTRACTION_MAX_CHARS", "4000"))
```

`TOP_K` (final chunks fed to LLM) stays at 4. `RETRIEVER_K = 10` widens the pre-fusion pool.

The hardcoded Azure API key in the current `config.py:8` is replaced with an env-only read (no fallback string in source).

---

## 6. Ingestion Pipeline

### 6.0 Chunk identity (used across all three indexes)

Every chunk gets a **deterministic ID** computed as:

```python
chunk_id = f"{doc_name}::{chunk_index}"
# example: "Sai_Prashanth_CV.pdf::3"
```

This ID is set as a metadata field on the LangChain `Document` during chunking (in the existing `DocumentProcessor.process_documents()` — one extra line: `chunk_metadata["chunk_id"] = f"{name}::{j}"`). All three indexes use the same ID:

- **Chroma**: stored as metadata; surfaced via `doc.metadata["chunk_id"]` on retrieval.
- **BM25**: stored alongside the tokenized text in the pickled corpus.
- **Neo4j**: used as the primary key for the `Chunk` node (`MERGE (c:Chunk {id: $chunk_id})`).

This makes RRF dedup trivial: identical `chunk_id` across retrievers = same chunk.

### 6.1 Flow per document

1. **Chunk** — existing `DocumentProcessor.process_file()` produces `List[Document]`.
2. **Vector index** — existing `vectorstore.add_documents(chunks)` writes to Chroma.
3. **BM25 index** — `BM25Retriever.add_documents(chunks)` tokenizes, appends to corpus, pickles to `./bm25_index/bm25.pkl`.
4. **Graph: chunk nodes** — `GraphStore.add_chunks(chunks)` runs `MERGE (c:Chunk {id, text, doc_name, chunk_index})`.
5. **Graph: entity extraction** — `GraphExtractor.extract_and_store(chunks, graph)` makes one gpt-4o call per chunk (concurrent up to `EXTRACTION_CONCURRENCY`), parses JSON response, writes entities/relations to Neo4j.

### 6.2 Entity-extraction prompt

```
Extract entities and relationships from the text below.

Entity types (use ONLY these):
  Person, Organization, Location, Concept, Event, Product, Technology, Date

Return JSON only:
{
  "entities": [{"name": "...", "type": "Person"}, ...],
  "relations": [{"source": "...", "target": "...", "type": "VERB_PHRASE"}, ...]
}

Text: {chunk_text}
```

### 6.3 Cypher writes

```cypher
// Chunk node (one per chunk)
MERGE (c:Chunk {id: $chunk_id})
SET c.text = $text, c.doc_name = $doc_name, c.chunk_index = $idx

// Entity (deduped by lowercased name)
MERGE (e:Entity {name_lower: toLower($name)})
SET e.name = $name, e.type = $type

// Mention edge
MERGE (e)-[:MENTIONED_IN]->(c)

// Relation between entities
MERGE (a:Entity {name_lower: toLower($source)})
MERGE (b:Entity {name_lower: toLower($target)})
MERGE (a)-[r:RELATION {type: $rel_type}]->(b)
```

`MERGE` (not `CREATE`) makes ingestion idempotent — re-ingesting the same document does not duplicate nodes or edges.

### 6.4 Cost estimate

Entity extraction costs ~1 gpt-4o call per chunk. A 50-page PDF ≈ 150 chunks ≈ ~$0.50–$1.00. Mitigated by `asyncio.gather` with semaphore-limited concurrency (default 5).

### 6.5 Failure handling during ingestion

| Failure | Behavior |
|---------|----------|
| Neo4j unreachable | Log warning, skip graph writes, continue with vector + BM25. |
| gpt-4o JSON malformed | Retry once with stricter "JSON only" reminder; on second failure, skip that chunk's entities. Document still queryable. |
| Chunk exceeds `EXTRACTION_MAX_CHARS` | Truncate to limit before sending. |
| Pickle write fails (BM25) | Raise — disk issues should be loud. |

---

## 7. Query Pipeline

### 7.1 Flow per question

1. **Fan-out (parallel via `asyncio.gather`)**:
   - Vector: `Chroma.as_retriever(k=RETRIEVER_K)` — 10 chunks.
   - BM25: tokenize question, `bm25.get_top_n(tokens, corpus, n=RETRIEVER_K)` — 10 chunks.
   - Graph: extract entities from question (1 gpt-4o call), match entities in Neo4j, traverse `GRAPH_HOPS` hops to retrieve chunks ranked by entity-match count.
2. **Fuse** via RRF — `top_k = TOP_K` (default 4) chunks remain.
3. **Generate** — existing prompt template + gpt-4o + StrOutputParser (LCEL chain in `rag_system.py:92-100`, swapped to use `HybridRetriever` instead of single Chroma retriever).
4. **Log** — existing MySQL logging via `MySQLLogger.log_query()`.

### 7.2 Graph retrieval — Cypher queries

**Step 1 — entity match in Neo4j**:
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) IN $entity_names_lower
   OR any(name IN $entity_names_lower WHERE toLower(e.name) CONTAINS name)
RETURN e
```

**Step 2 — 1-hop chunk retrieval (default)**:
```cypher
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE id(e) IN $entity_ids
RETURN c.text AS text, c.doc_name AS doc_name,
       c.chunk_index AS chunk_index, count(*) AS relevance
ORDER BY relevance DESC
LIMIT $k
```

**Step 3 — 2-hop chunk retrieval (if `GRAPH_HOPS=2`)**:
```cypher
MATCH (e:Entity)-[r:RELATION]-(neighbor:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE id(e) IN $entity_ids
RETURN c.text AS text, c.doc_name AS doc_name,
       c.chunk_index AS chunk_index, count(DISTINCT neighbor) AS hops
ORDER BY hops DESC
LIMIT $k
```

### 7.3 RRF fusion

Standard formula (Cormack, Clarke, Buettcher 2009):

```python
def rrf_fuse(rankings: List[List[Chunk]], k: int = 60) -> List[Chunk]:
    scores = defaultdict(float)
    for ranking in rankings:                       # one ranking per retriever
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] += 1 / (k + rank + 1)
    return sorted(all_chunks, key=lambda c: -scores[c.id])
```

`k=60` is the published constant; configurable via `RRF_K`. Chunks are deduped by `chunk_id` (defined in §6.0) — guaranteed identical across all three retrievers.

### 7.4 Failure handling at query time

| Failure | Behavior |
|---------|----------|
| Neo4j down | Skip graph retriever, fuse only vector + BM25. |
| BM25 index empty / not yet built | Skip BM25 retriever, fuse only vector + graph. |
| Graph entity-extraction LLM call fails | Skip graph retriever for this query. |
| All three retrievers return nothing | Existing fallback message: "I couldn't find relevant information…" (`rag_system.py:167-177`). |

The retriever degrades gracefully — never blocks the query because of one component.

---

## 8. Orchestration in `rag_system.py`

### 8.1 Constructor changes

```python
def __init__(self, config: RAGConfig):
    # ... existing init unchanged ...

    self.bm25 = BM25Retriever(config) if config.BM25_ENABLED else None
    self.graph = GraphStore(config) if config.GRAPH_ENABLED else None
    self.graph_extractor = GraphExtractor(self.llm, config) if config.GRAPH_ENABLED else None

    self.hybrid_retriever = HybridRetriever(
        vector_retriever=self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        ),
        bm25_retriever=self.bm25,
        graph_retriever=self.graph,
        config=config,
    )
```

### 8.2 LCEL chain swap

```python
self.qa_chain = (
    {
        "context": self.hybrid_retriever | format_docs,   # was: self.retriever
        "question": RunnablePassthrough()
    }
    | PROMPT
    | self.llm
    | StrOutputParser()
)
```

### 8.3 Ingestion methods extended

`add_documents()`, `add_file()`, `add_directory()` each gain conditional BM25 + graph write blocks after the existing Chroma write. MySQL logging stays untouched.

---

## 9. Testing Strategy

### 9.1 Unit tests (no external services)

| File | Coverage |
|------|----------|
| `test_hybrid_rrf.py` | RRF math: standard input → known output; empty list; identical rankings; one retriever returning nothing. Pure function. |
| `test_bm25.py` | Tokenization, add/retrieve, pickle save/load roundtrip. |
| `test_extractor.py` | JSON parse success; malformed JSON → graceful skip; truncation respects `EXTRACTION_MAX_CHARS`. Mocks `AzureChatOpenAI`. |
| `test_graph_store.py` | Cypher query string construction. Uses Neo4j Python driver mock. |

Target: ~12–15 unit tests, run in <5 seconds via `pytest tests/ -v`.

### 9.2 Integration tests (Neo4j required)

Use `testcontainers-python` to spin up Neo4j:

| Test | Purpose |
|------|---------|
| `test_ingest_creates_entities` | Add a fixture doc → query Neo4j → assert entity nodes exist. |
| `test_graph_retriever_finds_chunks` | 2-doc setup → graph query → assert correct chunk returned. |
| `test_hybrid_e2e` | Full ingest + query, assert all 3 retrievers contribute. |

Skipped if Docker unavailable or `NEO4J_URI` not set.

### 9.3 Manual smoke test

`tests/smoke_test.py`: ingests `Sai_Prashanth_CV.pdf`, asks 4 representative questions covering each retriever's strength, prints answers and which retrievers contributed. Visual sanity check — not asserted.

### 9.4 Out of scope for testing

- gpt-4o output quality (LLM is non-deterministic).
- Chroma library internals.
- Network failures to Azure.
- MySQL operations (existing, unchanged).

---

## 10. Resilience & Feature Flags

`GRAPH_ENABLED` and `BM25_ENABLED` (env-controlled) let the system run in degraded modes:

| Mode | `GRAPH_ENABLED` | `BM25_ENABLED` | Behavior |
|------|-----------------|-----------------|----------|
| Full | true | true | Vector + BM25 + Graph + RRF |
| No-graph | false | true | Vector + BM25 + RRF (2-way) |
| No-BM25 | true | false | Vector + Graph + RRF (2-way) |
| Vector-only | false | false | Identical to current system (zero regression) |

Flags allow demoing on a laptop without Neo4j, debugging individual retrievers, and providing an off-switch if Neo4j or the BM25 pickle is corrupted.

---

## 11. Security Improvements

The current `config.py:8` hardcodes an Azure API key as a default. As part of this work:

- All credentials move to `.env` (gitignored).
- `config.py` reads `os.getenv(...)` with **no fallback string** for secrets.
- `.env.example` shows variable names with placeholder values, committed.
- `.gitignore` adds `.env` and `bm25_index/`.

---

## 12. Resume Bullet (Target)

> Built a production-grade Graph-RAG system in Python: hybrid retrieval combining ChromaDB dense embeddings, BM25 sparse retrieval, and a Neo4j knowledge graph (8-type entity schema with LLM-driven relation extraction), fused via Reciprocal Rank Fusion (Cormack et al. 2009). Features async entity extraction with semaphore-limited concurrency, graceful degradation when components are unavailable, MySQL session logging, and integration tests via Neo4j testcontainers.

---

## 13. Open Questions / Risks

| Risk | Mitigation |
|------|-----------|
| gpt-4o entity extraction is noisy (duplicate entities, spurious relations) | Lowercased-name dedup via `MERGE`. Acceptable for general-purpose; could add a post-processing pass later if quality is poor. |
| Cost of extraction at scale | Async parallelism. For very large corpora, future work could batch chunks per call. |
| Neo4j Aura free tier limits (50k nodes / 175k relations) | Sufficient for demo and small corpora. Document the limit. |
| BM25 pickle grows unbounded | Acceptable up to ~10k chunks (~tens of MB). Future work: shard or migrate to Whoosh/Tantivy. |
| Test flakiness from `testcontainers` cold start | Tests skip gracefully if Docker missing; CI uses container caching. |

---

## 14. Implementation Order (preview — full plan in next step)

1. Config + dotenv + secrets cleanup.
2. BM25 retriever + tests.
3. RRF fusion + tests.
4. Graph store (Neo4j connection + Cypher) + tests.
5. Graph extractor (LLM-based) + tests.
6. Hybrid retriever (parallel + RRF) + tests.
7. Wire into `rag_system.py`, extend ingestion methods.
8. Integration tests via testcontainers.
9. Smoke test against the real CV.
10. README updates documenting the new architecture.

The full step-by-step plan with code-level detail will be produced by the `writing-plans` skill in the next phase.
