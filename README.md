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
# Unit tests (fast, no external services) - 26 tests
pytest tests/ --ignore=tests/integration

# Integration tests (requires Docker for Neo4j testcontainer)
pytest tests/integration

# Manual smoke test (real Azure + Neo4j Aura)
python tests/smoke_test.py
```

## How retrieval modes degrade

| `GRAPH_ENABLED` | `BM25_ENABLED` | Behavior |
|-----------------|----------------|----------|
| true            | true           | Vector + BM25 + Graph + RRF |
| false           | true           | Vector + BM25 + RRF |
| true            | false          | Vector + Graph + RRF |
| false           | false          | Vector-only |

If Neo4j is unreachable at runtime, the graph retriever silently drops out and the other two continue.

## Spec & Plan

- Design spec: `docs/superpowers/specs/2026-05-06-graph-rag-hybrid-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-graph-rag-hybrid.md`
