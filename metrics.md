# Evaluation Metrics & Pipeline Performance Guide

This document is the **single source of truth** for evaluating this Graph RAG system. It covers what to measure, how to measure it, how to interpret results, where the system can break, and what targets to aim for.

Use it as a checklist while running evaluation — start at the top, walk through each layer.

---

## Table of Contents

1. [The 4-Layer Evaluation Model](#the-4-layer-evaluation-model)
2. [Layer 1: Retrieval Metrics](#layer-1-retrieval-metrics)
3. [Layer 2: Generation Metrics (RAGAS)](#layer-2-generation-metrics-ragas)
4. [Layer 3: End-to-End Metrics](#layer-3-end-to-end-metrics)
5. [Layer 4: Operational Metrics](#layer-4-operational-metrics)
6. [Hybrid-Specific Metrics](#hybrid-specific-metrics)
7. [Step-by-Step Evaluation Pipeline](#step-by-step-evaluation-pipeline)
8. [Building the Gold-Standard Dataset](#building-the-gold-standard-dataset)
9. [End-to-End Pipeline Performance Benchmarking](#end-to-end-pipeline-performance-benchmarking)
10. [Where the System Breaks (Failure Modes)](#where-the-system-breaks-failure-modes)
11. [Interpretation Guide — What Scores Mean](#interpretation-guide--what-scores-mean)
12. [Resume Targets — Numbers Worth Putting on a CV](#resume-targets--numbers-worth-putting-on-a-cv)

---

## The 4-Layer Evaluation Model

A RAG system has 4 layers of evaluation, each answering a different question:

```
+-----------------------------------------------------------+
| LAYER 4: Operational                                      |
|   "Is it fast enough? Does it scale? What does it cost?" |
+-----------------------------------------------------------+
| LAYER 3: End-to-End                                       |
|   "Did the user get the right answer overall?"            |
+-----------------------------------------------------------+
| LAYER 2: Generation                                       |
|   "Given the chunks, did the LLM answer well?"            |
+-----------------------------------------------------------+
| LAYER 1: Retrieval                                        |
|   "Did we fetch the right chunks?"                        |
+-----------------------------------------------------------+
```

**You must evaluate ALL four layers.** A common mistake is to test only end-to-end answers — that hides whether problems live in retrieval or generation. Per-layer evaluation tells you exactly where to fix.

---

## Layer 1: Retrieval Metrics

These measure quality of `vector + BM25 + graph + RRF` **independent of the LLM**.

### Required: Gold-standard dataset
For each test question, you need a ground-truth list of relevant `chunk_ids`. Build manually for ~30–50 questions over your corpus (see [Building the Gold-Standard Dataset](#building-the-gold-standard-dataset)).

### Metrics

| Metric | Formula | Range | What it tells you |
|--------|---------|-------|-------------------|
| **Precision@K** | (relevant chunks in top-K) / K | 0-1 | Of what we returned, how much was on-topic |
| **Recall@K** | (relevant chunks in top-K) / (total relevant) | 0-1 | Did we find everything that was relevant |
| **Hit Rate@K** | 1 if any relevant chunk in top-K else 0 | 0/1 | Did we find at least one good chunk |
| **MRR** (Mean Reciprocal Rank) | mean of 1/(rank of first relevant) | 0-1 | How early does the first good chunk appear |
| **NDCG@K** | weighted by relevance & position | 0-1 | Gold standard — penalizes burying relevant chunks deeper |

### NDCG@K — the most important retrieval metric

NDCG (Normalized Discounted Cumulative Gain) is the de-facto standard. Formula:

```
DCG@K  = sum_{i=1..K} (relevance_i / log2(i + 1))
IDCG@K = DCG of the IDEAL ranking (relevance sorted descending)
NDCG@K = DCG@K / IDCG@K
```

- A relevant chunk at rank 1 contributes more than at rank 4.
- Score of 1.0 = ranking matches ideal.
- Score of 0.5 = relevant chunks are present but buried.

### What to track for THIS project

Since `TOP_K = 4` (final chunks fed to LLM):

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Recall@4** | > 0.85 | At least 85% of relevant chunks should be in the 4 we send to gpt-4o |
| **MRR** | > 0.7 | First relevant chunk should usually be at rank 1 or 2 |
| **NDCG@4** | > 0.75 | Strong overall ranking quality |
| **Hit Rate@4** | > 0.95 | Almost every query should find at least one good chunk |

### Code skeleton

```python
def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(relevant_ids)) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(relevant_ids)) / len(relevant_ids)

def reciprocal_rank(retrieved_ids, relevant_ids):
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    import math
    dcg = sum(
        (1.0 / math.log2(i + 2))
        for i, cid in enumerate(retrieved_ids[:k]) if cid in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0
```

---

## Layer 2: Generation Metrics (RAGAS)

This is where the **RAGAS** framework comes in — `pip install ragas`. Each metric is computed by an LLM judge that reads the question, context, and answer.

### The four core RAGAS metrics

| Metric | What it measures | Inputs needed |
|--------|------------------|---------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucinations) | question, context, answer |
| **Answer Relevancy** | Does the answer actually address the question? | question, answer |
| **Context Precision** | Were retrieved chunks relevant to the question? | question, context, ground_truth |
| **Context Recall** | Did retrieval get everything needed for the ground-truth answer? | context, ground_truth |

### Faithfulness — the single most important RAG metric

Faithfulness directly measures **hallucination**. The LLM judge:
1. Extracts every factual claim from your answer.
2. For each claim, checks whether it can be inferred from the retrieved context.
3. Score = (verifiable claims) / (total claims).

A faithfulness < 0.8 means your system is making things up despite the "use ONLY the context" prompt.

### What to track

| Metric | Target | If below |
|--------|--------|----------|
| **Faithfulness** | > 0.90 | LLM is hallucinating; tighten prompt or improve retrieval |
| **Answer Relevancy** | > 0.85 | Answer is off-topic; check question understanding |
| **Context Precision** | > 0.75 | Retrieved chunks are irrelevant; tune retrieval, RRF, or graph |
| **Context Recall** | > 0.80 | Missing relevant chunks; raise `RETRIEVER_K`, improve graph |

### Code skeleton (RAGAS)

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)

# Build dataset from your eval runs
data = {
    "question":     [...],   # list of questions
    "contexts":     [...],   # list of list of strings (retrieved chunks)
    "answer":       [...],   # list of generated answers
    "ground_truth": [...],   # list of gold-standard answers
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=your_azure_chat_openai,        # for the judge
    embeddings=your_azure_embeddings,  # for similarity calcs
)
print(result)
```

---

## Layer 3: End-to-End Metrics

These tell you whether the *user* got the right answer.

| Metric | Description | When to use |
|--------|-------------|-------------|
| **Answer Correctness** (RAGAS) | Combines factual accuracy + semantic similarity to a gold answer | Open-ended Q&A — your case |
| **Answer Semantic Similarity** (RAGAS) | Embedding cosine between generated and gold answer | Quick proxy when you don't want LLM-judged correctness |
| **Exact Match / F1** | Strict NLP metrics | Short factoid answers ("What year?") |
| **Human Eval (Likert 1-5)** | Manual scoring | When you have stakeholders or need final sign-off |

For open-ended QA over docs, **RAGAS `answer_correctness` is the most honest single number**. Target > 0.80.

---

## Layer 4: Operational Metrics

These don't measure quality but are critical for production. Track them in MySQL — your `query_history` table already logs `execution_time` and `num_sources`.

| Metric | What to track | Target | Where to measure |
|--------|---------------|--------|------------------|
| **Latency p50** | median time per query | < 3s | `query_history.execution_time` |
| **Latency p95** | 95th percentile time | < 8s | same |
| **Cost per query** | Azure OpenAI tokens × pricing | track $/1k queries | LangSmith or manual instrumentation |
| **Ingestion cost** | gpt-4o calls for entity extraction | < $1 per 100 chunks | log per `extract_and_store` call |
| **Retrieval coverage** | % queries where ≥1 chunk found | > 95% | check `num_sources > 0` |
| **Graph contribution rate** | % of final TOP_K from graph (after RRF) | > 0% to justify graph layer | instrument `HybridRetriever.invoke` |
| **Throughput (QPS)** | concurrent queries per second | depends on Azure tier | load test |
| **Error rate** | % queries that crash/timeout | < 0.1% | exception logging |

### Latency budget breakdown (typical)

For your hybrid pipeline, here's where time goes:

```
| Stage                      | Time (typical) | Notes                          |
|----------------------------|----------------|--------------------------------|
| Question embedding         |  ~100 ms       | text-embedding-ada-002 call    |
| Vector search (Chroma)     |   ~50 ms       | local, very fast               |
| BM25 search                |   ~10 ms       | in-memory                      |
| Graph entity extraction    |  ~500-800 ms   | gpt-4o call (one per query)    |
| Graph Cypher query         |  ~100-300 ms   | network to Aura cloud          |
| RRF fusion                 |    ~5 ms       | pure Python                    |
| LLM generation             | ~1500-3000 ms  | gpt-4o, depends on output len  |
|----------------------------|----------------|--------------------------------|
| TOTAL p50                  |  ~2.3-4.2 s    |                                |
```

The graph entity-extraction call is your biggest fixable cost — caching common questions would help.

---

## Hybrid-Specific Metrics

These are the metrics that make a 3-retriever hybrid system look distinctive — vanilla RAG projects can't measure them.

### 1. Per-retriever contribution analysis

For each query, track: of the final top-4 chunks (after RRF), how many came from each retriever's top-10 list?

```
Example aggregate over 100 queries:
  Vector contributed: 62% of final chunks
  BM25 contributed:   25% of final chunks
  Graph contributed:  13% of final chunks
```

**Why it matters**: If graph contributes 0%, it's broken or unhelpful. If everything is vector, BM25 and graph are dead weight.

### 2. Ablation study (THE resume bullet)

Run the same eval dataset against **4 configurations** by toggling feature flags:

| Config | `BM25_ENABLED` | `GRAPH_ENABLED` |
|--------|----------------|-----------------|
| A. Vector only | false | false |
| B. Vector + BM25 | true | false |
| C. Vector + Graph | false | true |
| D. **Full hybrid** | true | true |

Plot NDCG@4 + Faithfulness + Context Recall for each. **The delta between A and D is your "I built something useful" proof.**

### 3. RRF tuning sweep

Vary `RRF_K` over `[10, 30, 60, 100, 200]` and re-run eval. Find the optimum for your corpus. (Usually 60 is fine — Cormack et al. 2009 — but the experiment itself looks rigorous.)

### 4. Graph hops ablation

Compare `GRAPH_HOPS=1` vs `GRAPH_HOPS=2`. Track:
- NDCG@4 (does 2-hop find more relevant chunks?)
- Latency (does 2-hop slow things down?)
- Faithfulness (does 2-hop introduce noise that confuses the LLM?)

Often 2-hop helps for **relational questions** but hurts overall faithfulness due to noise.

### 5. Chunk size sweep

Re-ingest with `CHUNK_SIZE` in `[500, 800, 1000, 1500, 2000]`. Smaller = more precise retrieval but loses context; larger = more context but noisier embeddings. Plot the curve.

---

## Step-by-Step Evaluation Pipeline

This is the actual workflow to run end-to-end evaluation.

### Step 1: Install RAGAS

```bash
pip install ragas datasets
```

Add to `requirements.txt`:
```
ragas>=0.2.0
datasets>=2.0.0
```

### Step 2: Build the gold dataset

Create `eval/gold_dataset.json` with 30-50 entries (see [next section](#building-the-gold-standard-dataset)).

### Step 3: Create `eval/run_eval.py`

This script:
1. Loads the gold dataset
2. For each question, runs the full RAG pipeline
3. Captures retrieved chunks + generated answer
4. Builds a HuggingFace `Dataset`
5. Runs RAGAS metrics
6. Outputs a JSON report + a markdown summary

Skeleton:

```python
import json
from dotenv import load_dotenv
load_dotenv()

from config import RAGConfig
from rag_system import DocumentRAGSystem

# Per-retriever instrumentation: capture which retriever produced each chunk
# (Add hooks in HybridRetriever.invoke to record this.)

def run_eval(gold_path, config_overrides=None):
    config = RAGConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    rag = DocumentRAGSystem(config)
    gold = json.load(open(gold_path))

    results = []
    for entry in gold:
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        relevant_ids = entry["relevant_chunk_ids"]

        retrieved = rag.hybrid_retriever.invoke(question)
        retrieved_ids = [d.metadata["chunk_id"] for d in retrieved]
        contexts = [d.page_content for d in retrieved]

        response = rag.query(question)
        answer = response["answer"]

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "execution_time": response["execution_time"],
        })

    rag.close()
    return results

if __name__ == "__main__":
    results = run_eval("eval/gold_dataset.json")
    json.dump(results, open("eval/raw_results.json", "w"), indent=2)
```

### Step 4: Compute Layer 1 metrics

```python
from your_metrics_module import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank

for r in results:
    r["precision@4"] = precision_at_k(r["retrieved_ids"], r["relevant_ids"], 4)
    r["recall@4"]    = recall_at_k(r["retrieved_ids"], r["relevant_ids"], 4)
    r["mrr"]         = reciprocal_rank(r["retrieved_ids"], r["relevant_ids"])
    r["ndcg@4"]      = ndcg_at_k(r["retrieved_ids"], r["relevant_ids"], 4)

# Aggregate
import statistics
print("NDCG@4 (mean):", statistics.mean(r["ndcg@4"] for r in results))
print("MRR     (mean):", statistics.mean(r["mrr"]    for r in results))
```

### Step 5: Compute Layer 2 metrics (RAGAS)

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

dataset = Dataset.from_dict({
    "question":     [r["question"]     for r in results],
    "contexts":     [r["contexts"]     for r in results],
    "answer":       [r["answer"]       for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})

ragas_result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=rag.llm,
    embeddings=rag.embeddings,
)
print(ragas_result)
```

### Step 6: Run the ablation study

```python
configs = {
    "A_vector_only":  {"BM25_ENABLED": False, "GRAPH_ENABLED": False},
    "B_vec_bm25":     {"BM25_ENABLED": True,  "GRAPH_ENABLED": False},
    "C_vec_graph":    {"BM25_ENABLED": False, "GRAPH_ENABLED": True},
    "D_full_hybrid":  {"BM25_ENABLED": True,  "GRAPH_ENABLED": True},
}

ablation = {}
for name, overrides in configs.items():
    results = run_eval("eval/gold_dataset.json", overrides)
    ablation[name] = compute_all_metrics(results)
    json.dump(ablation, open("eval/ablation.json", "w"), indent=2)
```

### Step 7: Generate the comparison table

Output a markdown table with all 4 configs × all metrics. Paste it into the README.

### Step 8: Plot results (optional but resume-worthy)

```python
import matplotlib.pyplot as plt

configs = list(ablation.keys())
faithfulness_scores = [ablation[c]["faithfulness"] for c in configs]
ndcg_scores        = [ablation[c]["ndcg@4"]       for c in configs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(configs)), faithfulness_scores, label="Faithfulness")
ax.bar(range(len(configs)), ndcg_scores, alpha=0.5, label="NDCG@4")
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, rotation=15)
ax.legend()
ax.set_title("Hybrid Retrieval Ablation")
plt.savefig("eval/ablation.png", dpi=150, bbox_inches="tight")
```

---

## Building the Gold-Standard Dataset

The whole evaluation pipeline depends on this. Don't skip it.

### Format

`eval/gold_dataset.json`:
```json
[
  {
    "question": "What programming languages does Sai know?",
    "ground_truth": "Sai knows Python, JavaScript, and SQL based on the CV.",
    "relevant_chunk_ids": [
      "Sai_Prashanth_CV.pdf::2",
      "Sai_Prashanth_CV.pdf::5"
    ],
    "category": "skills",
    "difficulty": "easy"
  }
]
```

### How to build it (manual but worth it)

1. **Ingest your corpus once.**
2. **Browse Neo4j Browser** (graph) and **inspect Chroma** (vector) — get a feel for what's actually in your indexes.
3. **Write 30-50 questions covering**:
   - Easy factoids: "What is X?"
   - Listing: "List all Y."
   - Relational: "Who is connected to Z?"
   - Synthesis: "Summarize the work history."
   - Acronyms / exact-match: "What does PMP stand for?" (BM25 stress test)
   - Out-of-corpus: "What is the capital of France?" (system should refuse / say "not in docs")
4. **For each question**, manually identify which chunks (by chunk_id) actually contain the answer. Use the `chatbot.py` interface or query Neo4j directly.
5. **Write the ground-truth answer** in your own words.
6. **Tag** with category + difficulty.

### Question distribution target (for ~50 questions)

| Category | Count | Why |
|----------|-------|-----|
| Factoids | 12 | baseline — vector RAG should ace these |
| Listing | 8 | tests recall |
| Relational | 10 | tests graph traversal |
| Synthesis | 8 | tests generation quality |
| Exact-match (acronyms, names) | 7 | tests BM25 |
| Out-of-corpus | 5 | tests refusal behavior |

### Using LLM to bootstrap (semi-automated)

You can have gpt-4o draft questions from your chunks, then **edit them by hand**. Don't fully trust LLM-generated questions — they tend to be too easy ("rewordings of the chunk").

```python
prompt = f"""
Read this chunk from a document and write 3 diverse questions:
- 1 factoid (easy)
- 1 listing or comparison (medium)
- 1 relational ("who is connected to..." style) (hard)

For each, also list which keywords from the chunk are necessary
to answer it.

Chunk: {chunk.page_content}
"""
```

Always sanity-check by running each question through the system before evaluating — if it crashes the system, the question is malformed.

---

## End-to-End Pipeline Performance Benchmarking

This is separate from quality metrics — it's about **speed, cost, and scalability**.

### Latency benchmarking

```python
import time
import statistics

def benchmark_latency(rag, questions, n_runs=3):
    timings = {"total": [], "retrieval": [], "generation": []}
    for q in questions * n_runs:
        t0 = time.time()
        retrieved = rag.hybrid_retriever.invoke(q)
        t1 = time.time()
        response = rag.query(q)
        t2 = time.time()
        timings["retrieval"].append(t1 - t0)
        timings["generation"].append(t2 - t1)
        timings["total"].append(t2 - t0)
    for k, vals in timings.items():
        print(f"{k}: p50={statistics.median(vals):.2f}s  "
              f"p95={statistics.quantiles(vals, n=20)[-1]:.2f}s")
```

Run this before and after any tuning change. Track in a `benchmarks.md` log.

### Cost benchmarking

Add a callback to count tokens via LangChain's `get_openai_callback`:

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = rag.query(question)
    print(f"Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
```

Track averages:
- Cost per query (gpt-4o is the dominant cost)
- Ingestion cost per 100 chunks (entity extraction)

### Scalability test

Vary corpus size and re-run eval:

| Corpus size | Ingestion time | Avg query latency | Recall@4 |
|-------------|----------------|-------------------|----------|
| 10 docs (~1500 chunks) | ? | ? | ? |
| 50 docs | ? | ? | ? |
| 100 docs | ? | ? | ? |

This reveals where your stack starts straining. Common breaking points:
- **Chroma** handles millions of vectors fine (HNSW index).
- **BM25** memory grows linearly — fine to ~50k chunks, painful beyond.
- **Neo4j Aura free tier** caps at 50k nodes / 175k relationships.

### Concurrency test (if exposing as API)

```python
import asyncio
import aiohttp

async def hammer(question, n=50):
    async def one(session):
        async with session.post("http://localhost:8000/query", json={"q": question}) as r:
            return await r.json()
    async with aiohttp.ClientSession() as sess:
        tasks = [one(sess) for _ in range(n)]
        await asyncio.gather(*tasks)
```

Currently your project is CLI-only, so skip this until you wrap it in FastAPI. But knowing the test exists is a resume signal.

---

## Where the System Breaks (Failure Modes)

Documenting failure modes is one of the strongest signals you understand the system. Here's what to look for and how to test each.

### F1. Retrieval misses the relevant chunk

**Symptoms**: System answers "I couldn't find relevant info" but the info IS in the corpus.
**Test**: Add the question to the gold set with the actual chunk_id. Check `Recall@4`.
**Causes**:
- Embedding model can't match question phrasing to chunk phrasing (synonyms, acronyms)
- Chunk is too small/large (cuts mid-sentence)
- Question phrased very differently from doc

**Mitigations**:
- Increase `RETRIEVER_K` (cast wider net)
- Try larger `CHUNK_SIZE`
- Add query expansion (LLM rephrases the question, run multiple)
- Add re-ranker (e.g., cross-encoder)

### F2. LLM hallucinates despite grounding

**Symptoms**: Faithfulness < 0.8. Answer contains facts not in retrieved chunks.
**Test**: Run RAGAS faithfulness on each question.
**Causes**:
- Prompt isn't strict enough
- LLM falls back to training data when context is ambiguous
- Retrieved chunks are wrong, so LLM invents

**Mitigations**:
- Tighten prompt: "If unsure, say 'I don't know'"
- Lower `TEMPERATURE` (already 0.3)
- Improve retrieval first — usually upstream issue

### F3. Graph extraction produces noisy / duplicate entities

**Symptoms**: Neo4j has nodes like "Python", "python programming", "Python language" — should be one.
**Test**: `MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'python' RETURN e.name` — count rows.
**Causes**:
- gpt-4o varies entity naming
- `MERGE (e:Entity {name_lower: toLower($name)})` only handles case, not variants

**Mitigations**:
- Add fuzzy-matching consolidation pass after ingestion
- Use embedding similarity to merge near-duplicates
- Stricter prompt with examples

### F4. RRF dominated by one retriever

**Symptoms**: Per-retriever contribution analysis shows one retriever providing 95% of results.
**Test**: Track contribution rates across 100 queries.
**Causes**:
- Other retrievers returning empty / irrelevant lists
- Score-distribution skew

**Mitigations**:
- Verify each retriever's top-10 individually for sample queries
- Adjust `RRF_K` to amplify lower-ranked entries
- Per-retriever weights (advanced; not in current implementation)

### F5. Graph retrieval finds nothing because entity extraction at query time fails

**Symptoms**: Graph contribution rate near 0%.
**Test**: For a known-graph-relevant question, manually check if `_extract_entity_names` returns anything.
**Causes**:
- gpt-4o couldn't identify entities in the question
- Entity names in question don't match graph (e.g. "the company" vs "Acme")

**Mitigations**:
- Improve question entity-extraction prompt with examples
- Co-reference resolution
- Lower the bar: retrieve any chunks mentioning question keywords as a fallback

### F6. Out-of-corpus questions still get answers

**Symptoms**: Ask "What's the capital of France?" → system answers "Paris" (from training data, not corpus).
**Test**: Include 5 out-of-corpus questions in the gold set with `ground_truth = "I cannot answer..."`.
**Causes**:
- Vector retriever returns *something* even when nothing relevant exists (cosine similarity is never zero)
- Prompt doesn't enforce hard refusal

**Mitigations**:
- Add a similarity threshold: if top chunk's score < threshold, refuse
- Add an explicit "say I don't know" instruction with examples
- Add a "verify the question is answerable from context" gate before generation

### F7. Latency spikes due to graph entity extraction

**Symptoms**: Average latency okay, p95 is bad.
**Test**: Track per-stage timing histograms.
**Causes**:
- gpt-4o entity extraction at query time is the slowest step (~500-800ms)
- Sometimes Azure throttles → 2-5s spikes

**Mitigations**:
- Cache entity-extraction results for repeated questions
- Use a smaller/faster model (gpt-4o-mini, claude-haiku) for query-time entity extraction
- Skip graph for short / keyword-y queries

### F8. Neo4j Aura free tier limits hit

**Symptoms**: Ingestion fails with quota error after 50k nodes.
**Test**: `MATCH (n) RETURN count(n)` in Neo4j Browser.
**Causes**:
- Free tier caps at 50k nodes / 175k relationships

**Mitigations**:
- Upgrade to paid tier
- Compact: dedupe entities more aggressively, prune low-frequency entities
- Switch to local Neo4j Community Edition (Docker)

### F9. Mysql connection drops mid-query

**Symptoms**: Query history occasionally not logged.
**Test**: Disconnect MySQL temporarily, run queries, check error logs.
**Causes**:
- `pymysql` doesn't reconnect on dropped connections
- Long idle periods cause server to close the connection

**Mitigations**:
- Wrap MySQL operations in retry logic
- Add `pymysql.connections.Connection.ping(reconnect=True)` before each query
- Switch to `mysql-connector-python` with built-in pooling

### F10. Asynchronous entity extraction overwhelms Azure rate limits

**Symptoms**: Ingestion fails with 429 errors mid-document.
**Test**: Ingest a 200-chunk doc with `EXTRACTION_CONCURRENCY=20`.
**Causes**:
- Azure has per-deployment TPM (tokens-per-minute) limits

**Mitigations**:
- Lower `EXTRACTION_CONCURRENCY` (default 5 is conservative)
- Add exponential backoff retry on 429
- Use Azure's built-in throttling responses

---

## Interpretation Guide — What Scores Mean

When you see a number, what does it actually mean?

### Faithfulness
- **0.95+**: Excellent — almost every claim is grounded
- **0.85-0.95**: Good — some minor inferences; acceptable
- **0.70-0.85**: Concerning — meaningful hallucinations happening
- **< 0.70**: Broken — the LLM is making things up. Fix prompt/retrieval before shipping.

### NDCG@4
- **0.90+**: Outstanding ranking
- **0.75-0.90**: Solid — what most production systems achieve
- **0.60-0.75**: OK — usable but improvable
- **< 0.60**: Retrieval is weak — investigate which retriever is dragging down

### Context Recall
- **0.90+**: We're finding nearly all the necessary info
- **0.70-0.90**: Acceptable — occasional misses
- **< 0.70**: Retrieval pool too narrow. Increase `RETRIEVER_K`, improve graph, or add re-ranker.

### Answer Correctness
- **0.85+**: Production-ready
- **0.70-0.85**: Beta-ready, needs eyes on each answer
- **< 0.70**: Not ready for users

### Latency
- **p95 < 5s**: Good UX
- **p95 5-10s**: Tolerable for batch / async use
- **p95 > 10s**: Real-time UX is impossible without caching/async

---

## Resume Targets — Numbers Worth Putting on a CV

If your evaluation produces these or better, you have something concrete to claim:

| Metric | Target for Resume |
|--------|-------------------|
| **NDCG@4** | > 0.80 |
| **Faithfulness** | > 0.90 |
| **Context Recall** | > 0.85 |
| **Answer Correctness** | > 0.80 |
| **Latency p95** | < 5 seconds |
| **Improvement vs vector-only baseline** | NDCG@4 +0.10 or more (the ablation study) |

### Resume bullet templates with numbers

> Built a Graph RAG hybrid retrieval system achieving **0.91 faithfulness** and **0.84 NDCG@4** on a 50-question evaluation set, a **+0.13 improvement** over vector-only baseline. Implemented Reciprocal Rank Fusion (Cormack et al. 2009) over ChromaDB dense embeddings, BM25 sparse retrieval, and Neo4j knowledge graph; reduced hallucinations by **25%** vs single-retriever baseline.

> Designed evaluation pipeline using RAGAS framework measuring 4 retrieval and 4 generation metrics across an ablation study of 4 configurations; documented 10 distinct failure modes and their mitigations.

These are dramatically more impressive than "I built a RAG."

---

## Suggested Project Structure for Eval Code

```
genai/
├── eval/
│   ├── __init__.py
│   ├── gold_dataset.json          # 30-50 questions with ground truth
│   ├── metrics.py                 # precision_at_k, ndcg_at_k, etc.
│   ├── run_eval.py                # main evaluation runner
│   ├── ablation.py                # 4-config ablation study
│   ├── benchmark.py               # latency / cost benchmarks
│   └── reports/
│       ├── ablation_2026-05-06.json
│       ├── ablation_2026-05-06.png
│       └── failure_analysis_2026-05-06.md
└── ...
```

---

## Final Checklist Before Claiming Evaluation is Done

- [ ] Built gold-standard dataset (30-50 questions across 6 categories)
- [ ] Layer 1 metrics computed: Precision@4, Recall@4, MRR, NDCG@4, Hit Rate@4
- [ ] Layer 2 metrics computed: Faithfulness, Answer Relevancy, Context Precision, Context Recall (RAGAS)
- [ ] Layer 3 metrics computed: Answer Correctness (RAGAS)
- [ ] Layer 4 metrics tracked: latency p50/p95, cost/query, retrieval coverage
- [ ] Per-retriever contribution analysis run (% from each retriever)
- [ ] Ablation study run (4 configs × all metrics)
- [ ] Hyperparameter sweeps run (RRF_K, CHUNK_SIZE, GRAPH_HOPS)
- [ ] All 10 failure modes tested or documented
- [ ] Results plotted and added to README
- [ ] Resume bullet updated with actual numbers

When all 11 boxes are checked, you have a defensible, rigorous, resume-grade evaluation.
