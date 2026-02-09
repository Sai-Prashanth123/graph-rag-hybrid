from typing import List, Dict, Any, Optional
import pymysql
import json
from datetime import datetime
from config import RAGConfig

class MySQLLogger:
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.connection = None
        self._setup_database()
    
    def _setup_database(self):
        try:
            conn = pymysql.connect(
                host=self.config.MYSQL_HOST,
                user=self.config.MYSQL_USER,
                password=self.config.MYSQL_PASSWORD,
                port=self.config.MYSQL_PORT
            )
            cursor = conn.cursor()
            
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.MYSQL_DATABASE}")
            cursor.close()
            conn.close()
            
            self.connection = pymysql.connect(
                host=self.config.MYSQL_HOST,
                user=self.config.MYSQL_USER,
                password=self.config.MYSQL_PASSWORD,
                database=self.config.MYSQL_DATABASE,
                port=self.config.MYSQL_PORT
            )
            
            cursor = self.connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    execution_time FLOAT,
                    num_sources INT,
                    session_id VARCHAR(255)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    document_name VARCHAR(255) NOT NULL,
                    document_type VARCHAR(50),
                    num_chunks INT,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_size INT,
                    file_path VARCHAR(500)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    total_queries INT DEFAULT 0
                )
            """)
            
            self.connection.commit()
            cursor.close()
            print("✓ MySQL database and tables initialized")
            
        except Exception as e:
            print(f"MySQL setup error: {e}")
            raise
    
    def log_query(self, query: str, response: str, context: List[str], 
                  execution_time: float, num_sources: int, session_id: Optional[str] = None):
        if self.connection is None:
            return
        try:
            cursor = self.connection.cursor()
            context_json = json.dumps(context)
            
            cursor.execute("""
                INSERT INTO query_history 
                (query, response, context, execution_time, num_sources, session_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (query, response, context_json, execution_time, num_sources, session_id))
            
            if session_id:
                cursor.execute("""
                    INSERT INTO conversation_sessions (session_id, last_activity, total_queries)
                    VALUES (%s, NOW(), 1)
                    ON DUPLICATE KEY UPDATE 
                        last_activity = NOW(),
                        total_queries = total_queries + 1
                """, (session_id,))
            
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging query: {e}")
    
    def log_document(self, doc_name: str, doc_type: str, num_chunks: int, 
                    file_size: int, file_path: Optional[str] = None):
        if self.connection is None:
            return
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO document_metadata 
                (document_name, document_type, num_chunks, file_size, file_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_name, doc_type, num_chunks, file_size, file_path))
            
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging document: {e}")
    
    def get_recent_queries(self, limit: int = 10, session_id: Optional[str] = None) -> List[Dict]:
        if self.connection is None:
            return []
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            if session_id:
                cursor.execute("""
                    SELECT query, response, timestamp, execution_time 
                    FROM query_history 
                    WHERE session_id = %s
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT query, response, timestamp, execution_time 
                    FROM query_history 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Error retrieving queries: {e}")
            return []
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        if self.connection is None:
            return []
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("""
                SELECT query, response, timestamp 
                FROM query_history 
                WHERE session_id = %s
                ORDER BY timestamp ASC 
                LIMIT %s
            """, (session_id, limit))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict]:
        if self.connection is None:
            return []
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("""
                SELECT document_name, document_type, num_chunks, 
                       upload_timestamp, file_size
                FROM document_metadata 
                ORDER BY upload_timestamp DESC
            """)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def close(self):
        if self.connection:
            self.connection.close()
