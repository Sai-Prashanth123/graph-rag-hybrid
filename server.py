from dotenv import load_dotenv
load_dotenv()  # Must load env before RAGConfig class body is parsed

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from config import RAGConfig
from kb_manager import KnowledgeBaseManager
from rag_system import DocumentRAGSystem


# ── Request / response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class UploadTextRequest(BaseModel):
    text: str
    name: str


class CreateKBRequest(BaseModel):
    name: str
    description: str = ""
    rag_type: str = "hybrid"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    retriever_k: int = 10
    graph_hops: int = 1


class UpdateKBRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rag_type: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    retriever_k: Optional[int] = None
    graph_hops: Optional[int] = None


class CreateAPIKeyRequest(BaseModel):
    name: str = "API Key"


# ── Prompt template ──────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are a helpful and knowledgeable assistant that answers questions based on the provided documents.

IMPORTANT INSTRUCTIONS:
1. Answer using ONLY the information provided in the context below.
2. If the question is a greeting, greet back and offer to help with the documents.
3. Extract and summarize relevant details from the context accurately.
4. If the context doesn't contain enough information, say what IS available and what's missing.
5. Be conversational and specific. Do not make up information.

Context from documents:
{context}

User Question: {question}

Answer (based on the context above):"""

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = RAGConfig()
    app.state.base_config = cfg
    app.state.kb_manager = KnowledgeBaseManager(cfg)
    app.state.rag_pool: Dict[str, DocumentRAGSystem] = {}
    app.state.rag_pool_lock = threading.Lock()
    print("FastAPI: B2B RAG Platform ready")
    yield
    with app.state.rag_pool_lock:
        for rag in app.state.rag_pool.values():
            try:
                rag.close()
            except Exception:
                pass
    print("FastAPI: All RAG systems closed")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Platform API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_config_for_kb(base: RAGConfig, kb: dict) -> RAGConfig:
    return RAGConfig(
        KB_ID=kb['kb_id'],
        COLLECTION_NAME=f"kb_{kb['kb_id'][:8]}",
        BM25_INDEX_PATH=f"./bm25_index/{kb['kb_id']}.pkl",
        CHUNK_SIZE=kb['chunk_size'],
        CHUNK_OVERLAP=kb['chunk_overlap'],
        TOP_K=kb['top_k'],
        RETRIEVER_K=kb['retriever_k'],
        GRAPH_HOPS=kb['graph_hops'],
        # Inherit API credentials
        GROQ_API_KEY=base.GROQ_API_KEY,
        GROQ_MODEL=base.GROQ_MODEL,
        HF_API_KEY=base.HF_API_KEY,
        HF_EMBEDDING_MODEL=base.HF_EMBEDDING_MODEL,
        CHROMA_PERSIST_DIR=base.CHROMA_PERSIST_DIR,
        MYSQL_HOST=base.MYSQL_HOST,
        MYSQL_USER=base.MYSQL_USER,
        MYSQL_PASSWORD=base.MYSQL_PASSWORD,
        MYSQL_DATABASE=base.MYSQL_DATABASE,
        MYSQL_PORT=base.MYSQL_PORT,
        NEO4J_URI=base.NEO4J_URI,
        NEO4J_USER=base.NEO4J_USER,
        NEO4J_PASSWORD=base.NEO4J_PASSWORD,
        NEO4J_DATABASE=base.NEO4J_DATABASE,
        GRAPH_ENABLED=base.GRAPH_ENABLED,
        BM25_ENABLED=base.BM25_ENABLED,
        TEMPERATURE=base.TEMPERATURE,
        MAX_TOKENS=base.MAX_TOKENS,
        RRF_K=base.RRF_K,
    )


def _get_or_create_rag(kb_id: str) -> DocumentRAGSystem:
    with app.state.rag_pool_lock:
        if kb_id not in app.state.rag_pool:
            kb = app.state.kb_manager.get_kb(kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
            # Ensure bm25_index directory exists
            Path("./bm25_index").mkdir(exist_ok=True)
            config = _build_config_for_kb(app.state.base_config, kb)
            app.state.rag_pool[kb_id] = DocumentRAGSystem(config, rag_type=kb['rag_type'])
        return app.state.rag_pool[kb_id]


def _evict_rag(kb_id: str):
    with app.state.rag_pool_lock:
        if kb_id in app.state.rag_pool:
            try:
                app.state.rag_pool[kb_id].close()
            except Exception:
                pass
            del app.state.rag_pool[kb_id]


async def _api_key_auth(authorization: str = Header(...)) -> dict:
    """FastAPI dependency for Bearer token auth on /v1 endpoints."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must use Bearer scheme")
    api_key = authorization[7:].strip()
    info = app.state.kb_manager.validate_api_key(api_key)
    if not info:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    app.state.kb_manager.record_usage(api_key)
    return info  # {kb_id, key_id, kb_name}


async def _stream_response(rag: DocumentRAGSystem, question: str, session_id: str):
    """Shared SSE generator used by both internal and external chat/stream endpoints."""
    loop = asyncio.get_event_loop()
    start = time.time()
    kb_id = rag.config.KB_ID or None

    try:
        sources = await loop.run_in_executor(
            None, rag.hybrid_retriever.invoke, question
        )
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    if not sources:
        msg = "I couldn't find relevant information in the uploaded documents. Please upload documents first, then ask questions about their content."
        yield f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"
        exec_time = time.time() - start
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'session_id': session_id, 'execution_time': exec_time})}\n\n"
        return

    context = "\n\n".join(doc.page_content for doc in sources)
    prompt_text = _PROMPT_TEMPLATE.format(context=context, question=question)
    messages = [HumanMessage(content=prompt_text)]

    full_answer = ""
    async for chunk in rag.llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    exec_time = time.time() - start

    context_texts = [doc.page_content for doc in sources]
    await loop.run_in_executor(
        None,
        rag.logger.log_query,
        question, full_answer, context_texts, exec_time, len(sources), session_id, kb_id,
    )

    source_list = [
        {
            "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "metadata": doc.metadata,
        }
        for doc in sources
    ]
    yield f"data: {json.dumps({'type': 'done', 'sources': source_list, 'session_id': session_id, 'execution_time': exec_time})}\n\n"


def _sse_response(generator):
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/health ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    base_config = app.state.base_config
    sqlite_ok = False
    try:
        import sqlite3
        conn = sqlite3.connect(base_config.SQLITE_PATH)
        conn.execute("SELECT 1")
        conn.close()
        sqlite_ok = True
    except Exception:
        pass

    with app.state.rag_pool_lock:
        active_kbs = len(app.state.rag_pool)

    return {
        "status": "healthy",
        "active_kbs": active_kbs,
        "components": {
            "groq_llm": bool(base_config.GROQ_API_KEY),
            "chromadb": True,
            "bm25": base_config.BM25_ENABLED,
            "neo4j": base_config.GRAPH_ENABLED and bool(base_config.NEO4J_URI),
            "mysql": sqlite_ok,
        },
    }


# ── /api/kb — Knowledge Base CRUD ────────────────────────────────────────────

@app.post("/api/kb", status_code=201)
def create_kb(request: CreateKBRequest):
    valid_types = {'vector', 'bm25', 'hybrid', 'graph', 'full_hybrid'}
    if request.rag_type not in valid_types:
        raise HTTPException(400, f"Invalid rag_type. Must be one of: {', '.join(valid_types)}")
    kb = app.state.kb_manager.create_kb(
        name=request.name,
        description=request.description,
        rag_type=request.rag_type,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        top_k=request.top_k,
        retriever_k=request.retriever_k,
        graph_hops=request.graph_hops,
    )
    return kb


@app.get("/api/kb")
def list_kbs():
    return app.state.kb_manager.list_kbs()


@app.get("/api/kb/{kb_id}")
def get_kb(kb_id: str):
    kb = app.state.kb_manager.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    return kb


@app.patch("/api/kb/{kb_id}")
def update_kb(kb_id: str, request: UpdateKBRequest):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    kb = app.state.kb_manager.update_kb(kb_id, **updates)
    # Evict cached RAGSystem so next request re-initialises with new config
    _evict_rag(kb_id)
    return kb


@app.delete("/api/kb/{kb_id}", status_code=204)
def delete_kb(kb_id: str):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    _evict_rag(kb_id)
    # Drop chroma collection for this KB
    try:
        import chromadb
        base = app.state.base_config
        client = chromadb.PersistentClient(path=base.CHROMA_PERSIST_DIR)
        client.delete_collection(f"kb_{kb_id[:8]}")
    except Exception:
        pass
    app.state.kb_manager.delete_kb(kb_id)


# ── /api/kb/{kb_id}/upload ────────────────────────────────────────────────────

@app.post("/api/kb/{kb_id}/upload/file")
async def upload_file_to_kb(
    kb_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    rag = _get_or_create_rag(kb_id)
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        meta = {
            "name": file.filename,
            "type": suffix[1:] if suffix else "text",
            "file_size": len(content),
            "file_path": file.filename,
        }

        chunks = rag.add_file_to_stores(tmp_path, meta)

        if rag.graph is not None:
            background_tasks.add_task(rag.add_chunks_to_graph, chunks)

        # Update KB stats
        app.state.kb_manager.increment_kb_stats(kb_id, doc_delta=1, chunk_delta=len(chunks))

        graph_note = " Graph entity extraction running in background." if rag.graph else ""
        return {
            "success": True,
            "doc_name": file.filename,
            "num_chunks": len(chunks),
            "message": f"{len(chunks)} chunks indexed.{graph_note}",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/kb/{kb_id}/upload/text")
def upload_text_to_kb(kb_id: str, request: UploadTextRequest):
    if not request.text.strip():
        raise HTTPException(400, "Text content is empty.")
    rag = _get_or_create_rag(kb_id)
    meta = [{"name": request.name, "type": "text", "file_size": len(request.text.encode())}]
    rag.add_documents([request.text], meta)
    return {"success": True, "name": request.name}


# ── /api/kb/{kb_id}/documents & history ──────────────────────────────────────

@app.get("/api/kb/{kb_id}/documents")
def get_kb_documents(kb_id: str):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    from logger import MySQLLogger
    logger = MySQLLogger(app.state.base_config)
    docs = logger.get_all_documents(kb_id=kb_id)
    for doc in docs:
        if hasattr(doc.get("upload_timestamp"), "isoformat"):
            doc["upload_timestamp"] = doc["upload_timestamp"].isoformat()
    return docs


@app.get("/api/kb/{kb_id}/history")
def get_kb_history(kb_id: str, session_id: Optional[str] = None, limit: int = 20):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    from logger import MySQLLogger
    logger = MySQLLogger(app.state.base_config)
    rows = logger.get_recent_queries(limit=limit, session_id=session_id, kb_id=kb_id)
    for row in rows:
        if hasattr(row.get("timestamp"), "isoformat"):
            row["timestamp"] = row["timestamp"].isoformat()
    return rows


# ── /api/kb/{kb_id}/chat ─────────────────────────────────────────────────────

@app.post("/api/kb/{kb_id}/chat")
def kb_chat(kb_id: str, request: ChatRequest):
    rag = _get_or_create_rag(kb_id)
    session_id = request.session_id or str(uuid.uuid4())
    result = rag.query(request.question, session_id)
    # Serialise any datetime objects in sources metadata
    for src in result.get("sources", []):
        meta = src.get("metadata", {})
        for k, v in meta.items():
            if hasattr(v, "isoformat"):
                meta[k] = v.isoformat()
    return {**result, "session_id": session_id}


@app.post("/api/kb/{kb_id}/chat/stream")
async def kb_chat_stream(kb_id: str, request: ChatRequest):
    rag = _get_or_create_rag(kb_id)
    session_id = request.session_id or str(uuid.uuid4())
    return _sse_response(_stream_response(rag, request.question, session_id))


# ── /api/kb/{kb_id}/keys — API Key management ────────────────────────────────

@app.post("/api/kb/{kb_id}/keys", status_code=201)
def create_api_key(kb_id: str, request: CreateAPIKeyRequest):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    return app.state.kb_manager.create_api_key(kb_id, request.name)


@app.get("/api/kb/{kb_id}/keys")
def list_api_keys(kb_id: str):
    if not app.state.kb_manager.get_kb(kb_id):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")
    return app.state.kb_manager.list_api_keys(kb_id)


@app.delete("/api/kb/{kb_id}/keys/{key_id}", status_code=204)
def revoke_api_key(kb_id: str, key_id: str):
    revoked = app.state.kb_manager.revoke_api_key(key_id)
    if not revoked:
        raise HTTPException(404, f"API key '{key_id}' not found")


# ── /v1 — External embeddable API (Bearer token auth) ────────────────────────

@app.post("/v1/chat")
def v1_chat(request: ChatRequest, auth: dict = Depends(_api_key_auth)):
    kb_id = auth["kb_id"]
    rag = _get_or_create_rag(kb_id)
    session_id = request.session_id or str(uuid.uuid4())
    result = rag.query(request.question, session_id)
    for src in result.get("sources", []):
        meta = src.get("metadata", {})
        for k, v in meta.items():
            if hasattr(v, "isoformat"):
                meta[k] = v.isoformat()
    return {**result, "session_id": session_id, "kb_name": auth.get("kb_name")}


@app.post("/v1/chat/stream")
async def v1_chat_stream(request: ChatRequest, auth: dict = Depends(_api_key_auth)):
    kb_id = auth["kb_id"]
    rag = _get_or_create_rag(kb_id)
    session_id = request.session_id or str(uuid.uuid4())
    return _sse_response(_stream_response(rag, request.question, session_id))


@app.get("/v1/documents")
def v1_documents(auth: dict = Depends(_api_key_auth)):
    kb_id = auth["kb_id"]
    from logger import MySQLLogger
    logger = MySQLLogger(app.state.base_config)
    docs = logger.get_all_documents(kb_id=kb_id)
    for doc in docs:
        if hasattr(doc.get("upload_timestamp"), "isoformat"):
            doc["upload_timestamp"] = doc["upload_timestamp"].isoformat()
    return docs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
