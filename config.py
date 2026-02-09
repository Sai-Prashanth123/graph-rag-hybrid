import os
from typing import Optional


class RAGConfig:
    
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://agentplus.openai.azure.com/")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "1TPW16ifwPJccSiQPSHq63nU7IcT6R9DrduIHBYwCm5jbUWiSbkLJQQJ99BDACYeBjFXJ3w3AAABACOG3Yia")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_qa")
    
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_system")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    TOP_K = int(os.getenv("TOP_K", "4"))
    
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
