import sqlite3
import json
from typing import List, Dict, Any, Optional

from config import RAGConfig


class MySQLLogger:
    """Persistent log store backed by SQLite (no server required)."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.connection = None  # kept for API compat; always None at runtime
        self._setup_database()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open and return a fresh SQLite connection with row dict factory."""
        conn = sqlite3.connect(self.config.SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _setup_database(self):
        try:
            conn = self._connect()
            c = conn.cursor()

            c.execute("""
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
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    document_type TEXT,
                    num_chunks INTEGER,
                    upload_timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
                    file_size INTEGER,
                    file_path TEXT,
                    kb_id TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                    last_activity TEXT DEFAULT (CURRENT_TIMESTAMP),
                    total_queries INTEGER DEFAULT 0
                )
            """)

            c.execute("""
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
                )
            """)

            c.execute("""
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
                )
            """)

            # Idempotent: add kb_id column if absent
            for table in ('document_metadata', 'query_history'):
                existing = {row['name'] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}
                if 'kb_id' not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN kb_id TEXT")

            conn.commit()
            conn.close()
            print("✓ SQLite database and tables initialized")
        except Exception as e:
            print(f"SQLite setup error: {e}")
            raise

    # ── Write operations ────────────────────────────────────────────────────

    def log_query(self, query: str, response: str, context: List[str],
                  execution_time: float, num_sources: int,
                  session_id: Optional[str] = None,
                  kb_id: Optional[str] = None):
        try:
            conn = self._connect()
            c = conn.cursor()
            context_json = json.dumps(context)

            c.execute("""
                INSERT INTO query_history
                (query, response, context, execution_time, num_sources, session_id, kb_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (query, response, context_json, execution_time, num_sources, session_id, kb_id))

            if session_id:
                c.execute("""
                    INSERT INTO conversation_sessions (session_id, last_activity, total_queries)
                    VALUES (?, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_activity = CURRENT_TIMESTAMP,
                        total_queries = total_queries + 1
                """, (session_id,))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging query: {e}")

    def log_document(self, doc_name: str, doc_type: str, num_chunks: int,
                     file_size: int, file_path: Optional[str] = None,
                     kb_id: Optional[str] = None):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO document_metadata
                (document_name, document_type, num_chunks, file_size, file_path, kb_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_name, doc_type, num_chunks, file_size, file_path, kb_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging document: {e}")

    # ── Read operations ─────────────────────────────────────────────────────

    def get_recent_queries(self, limit: int = 10,
                           session_id: Optional[str] = None,
                           kb_id: Optional[str] = None) -> List[Dict]:
        try:
            conn = self._connect()
            c = conn.cursor()
            if session_id and kb_id:
                rows = c.execute("""
                    SELECT query, response, timestamp, execution_time
                    FROM query_history WHERE session_id=? AND kb_id=?
                    ORDER BY timestamp DESC LIMIT ?
                """, (session_id, kb_id, limit)).fetchall()
            elif session_id:
                rows = c.execute("""
                    SELECT query, response, timestamp, execution_time
                    FROM query_history WHERE session_id=?
                    ORDER BY timestamp DESC LIMIT ?
                """, (session_id, limit)).fetchall()
            elif kb_id:
                rows = c.execute("""
                    SELECT query, response, timestamp, execution_time
                    FROM query_history WHERE kb_id=?
                    ORDER BY timestamp DESC LIMIT ?
                """, (kb_id, limit)).fetchall()
            else:
                rows = c.execute("""
                    SELECT query, response, timestamp, execution_time
                    FROM query_history ORDER BY timestamp DESC LIMIT ?
                """, (limit,)).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            print(f"Error retrieving queries: {e}")
            return []

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        try:
            conn = self._connect()
            c = conn.cursor()
            rows = c.execute("""
                SELECT query, response, timestamp FROM query_history
                WHERE session_id=? ORDER BY timestamp ASC LIMIT ?
            """, (session_id, limit)).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Dict]:
        try:
            conn = self._connect()
            c = conn.cursor()
            if kb_id:
                rows = c.execute("""
                    SELECT document_name, document_type, num_chunks,
                           upload_timestamp, file_size
                    FROM document_metadata WHERE kb_id=?
                    ORDER BY upload_timestamp DESC
                """, (kb_id,)).fetchall()
            else:
                rows = c.execute("""
                    SELECT document_name, document_type, num_chunks,
                           upload_timestamp, file_size
                    FROM document_metadata ORDER BY upload_timestamp DESC
                """).fetchall()
            result = [dict(r) for r in rows]
            conn.close()
            return result
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def close(self):
        pass  # connection-per-operation — nothing persistent to close
