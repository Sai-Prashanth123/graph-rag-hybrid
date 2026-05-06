import os
from typing import List, Optional


class RAGConfig:
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_qa")

    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_system")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    TOP_K = int(os.getenv("TOP_K", "4"))

    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() == "true"

    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./bm25_index/bm25.pkl")
    BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"

    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))

    ENTITY_TYPES: List[str] = [
        "Person", "Organization", "Location", "Concept",
        "Event", "Product", "Technology", "Date",
    ]
    EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "5"))
    EXTRACTION_MAX_CHARS = int(os.getenv("EXTRACTION_MAX_CHARS", "4000"))
