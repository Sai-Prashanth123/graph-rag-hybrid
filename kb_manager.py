import secrets
import sqlite3
import uuid
from typing import Dict, List, Optional

from config import RAGConfig


class KnowledgeBaseManager:
    """CRUD for knowledge_bases and api_keys tables backed by SQLite."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_schema()

    def _setup_schema(self):
        """Ensure all required tables exist (idempotent)."""
        try:
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    rag_type TEXT DEFAULT 'hybrid',
                    chunk_size INTEGER DEFAULT 1000,
                    chunk_overlap INTEGER DEFAULT 200,
                    top_k INTEGER DEFAULT 4,
                    retriever_k INTEGER DEFAULT 10,
                    graph_hops INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                    updated_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                    total_documents INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT UNIQUE NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    kb_id TEXT NOT NULL,
                    name TEXT DEFAULT 'API Key',
                    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                    last_used TEXT,
                    total_requests INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    document_type TEXT,
                    num_chunks INTEGER,
                    upload_timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
                    file_size INTEGER,
                    file_path TEXT,
                    kb_id TEXT
                );
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT,
                    timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
                    execution_time REAL,
                    num_sources INTEGER,
                    session_id TEXT,
                    kb_id TEXT
                );
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                    last_activity TEXT DEFAULT (CURRENT_TIMESTAMP),
                    total_queries INTEGER DEFAULT 0
                );
            """)
            conn.close()
        except Exception as e:
            print(f"KBManager schema setup error: {e}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── Knowledge Base CRUD ─────────────────────────────────────────────────

    def create_kb(self, name: str, description: str = "",
                  rag_type: str = "hybrid",
                  chunk_size: int = 1000, chunk_overlap: int = 200,
                  top_k: int = 4, retriever_k: int = 10,
                  graph_hops: int = 1) -> Dict:
        kb_id = str(uuid.uuid4())
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO knowledge_bases
                (kb_id, name, description, rag_type, chunk_size, chunk_overlap,
                 top_k, retriever_k, graph_hops)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (kb_id, name, description, rag_type, chunk_size, chunk_overlap,
                  top_k, retriever_k, graph_hops))
            conn.commit()
        finally:
            conn.close()
        return self.get_kb(kb_id)

    def get_kb(self, kb_id: str) -> Optional[Dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM knowledge_bases WHERE kb_id = ?", (kb_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def list_kbs(self) -> List[Dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM knowledge_bases ORDER BY created_at DESC"
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def update_kb(self, kb_id: str, **fields) -> Optional[Dict]:
        allowed = {'name', 'description', 'rag_type', 'chunk_size', 'chunk_overlap',
                   'top_k', 'retriever_k', 'graph_hops'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_kb(kb_id)
        # Always bump updated_at
        updates['updated_at'] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [kb_id]
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE knowledge_bases SET {set_clause} WHERE kb_id = ?", vals
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_kb(kb_id)

    def delete_kb(self, kb_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,)
            )
            deleted = cur.rowcount > 0
            conn.commit()
        finally:
            conn.close()
        return deleted

    def increment_kb_stats(self, kb_id: str, doc_delta: int = 0, chunk_delta: int = 0):
        if not doc_delta and not chunk_delta:
            return
        conn = self._connect()
        try:
            conn.execute("""
                UPDATE knowledge_bases
                SET total_documents = total_documents + ?,
                    total_chunks = total_chunks + ?,
                    updated_at = ?
                WHERE kb_id = ?
            """, (doc_delta, chunk_delta, _now(), kb_id))
            conn.commit()
        finally:
            conn.close()

    # ── API Key operations ──────────────────────────────────────────────────

    def create_api_key(self, kb_id: str, name: str = "API Key") -> Dict:
        key_id = str(uuid.uuid4())
        api_key = f"rag-{kb_id[:8]}-{secrets.token_hex(12)}"
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO api_keys (key_id, api_key, kb_id, name)
                VALUES (?, ?, ?, ?)
            """, (key_id, api_key, kb_id, name))
            conn.commit()
        finally:
            conn.close()
        return {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "kb_id": kb_id,
            "is_active": True,
            "total_requests": 0,
        }

    def list_api_keys(self, kb_id: str) -> List[Dict]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT key_id,
                       substr(api_key, 1, 16) || '****' AS api_key_preview,
                       name, kb_id, created_at, last_used, total_requests, is_active
                FROM api_keys
                WHERE kb_id = ?
                ORDER BY created_at DESC
            """, (kb_id,)).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def revoke_api_key(self, key_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
            updated = cur.rowcount > 0
            conn.commit()
        finally:
            conn.close()
        return updated

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT ak.kb_id, ak.key_id, kb.name AS kb_name
                FROM api_keys ak
                JOIN knowledge_bases kb ON ak.kb_id = kb.kb_id
                WHERE ak.api_key = ? AND ak.is_active = 1
            """, (api_key,)).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def record_usage(self, api_key: str):
        conn = self._connect()
        try:
            conn.execute("""
                UPDATE api_keys
                SET last_used = ?, total_requests = total_requests + 1
                WHERE api_key = ?
            """, (_now(), api_key))
            conn.commit()
        finally:
            conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
