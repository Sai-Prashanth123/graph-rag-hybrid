from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import chromadb
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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

# RAG type routing sets
_VECTOR_TYPES = {'vector', 'hybrid', 'full_hybrid'}
_BM25_TYPES   = {'bm25',   'hybrid', 'full_hybrid'}
_GRAPH_TYPES  = {'graph',            'full_hybrid'}


class DocumentRAGSystem:

    def __init__(self, config: RAGConfig, rag_type: str = 'full_hybrid'):
        self.config = config
        self.rag_type = rag_type
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self.logger = MySQLLogger(config)
        self.document_processor = DocumentProcessor(config)

        self._initialize_components()

    def _initialize_components(self):
        print(f"Initializing RAG System (type={self.rag_type})…")

        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.config.HF_EMBEDDING_MODEL,
            huggingfacehub_api_token=self.config.HF_API_KEY,
        )
        print("✓ Embeddings initialized (HuggingFace)")

        self.llm = ChatGroq(
            model=self.config.GROQ_MODEL,
            api_key=self.config.GROQ_API_KEY,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
        )
        print("✓ LLM initialized (Groq)")

        # Vector store — only if this rag_type needs it
        self.vectorstore = None
        self.retriever = None
        if self.rag_type in _VECTOR_TYPES:
            chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIR
            )
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.RETRIEVER_K}
            )
            print("✓ Vector store initialized")

        # BM25 — only if this rag_type needs it
        self.bm25 = None
        if self.rag_type in _BM25_TYPES and self.config.BM25_ENABLED:
            try:
                self.bm25 = BM25Retriever(self.config)
                print("✓ BM25 retriever initialized")
            except Exception as e:
                print(f"Warning: BM25 disabled ({e})")

        # Graph — only if this rag_type needs it
        self.graph = None
        self.graph_extractor = None
        if self.rag_type in _GRAPH_TYPES and self.config.GRAPH_ENABLED:
            try:
                kb_id = self.config.KB_ID or 'default'
                self.graph = GraphStore(self.config, kb_id=kb_id)
                self.graph_extractor = GraphExtractor(self.llm, self.config)
                print("✓ Neo4j graph store initialized")
            except Exception as e:
                print(f"Warning: graph disabled ({e})")
                self.graph = None
                self.graph_extractor = None

        prompt_template = """You are a helpful and knowledgeable assistant that answers questions based on the provided documents. Your role is to extract and present information from the context documents accurately and clearly.

IMPORTANT INSTRUCTIONS:
1. Answer questions using ONLY the information provided in the context below.
2. If the question is a greeting (like "hi", "hello"), greet back and offer to help answer questions about the documents.
3. For questions about people, places, skills, experiences, or information mentioned in the documents, extract and summarize the relevant details from the context.
4. If the context doesn't contain enough information to fully answer the question, say what information IS available and mention what's missing.
5. Be conversational, helpful, and provide specific details from the documents when available.
6. Don't make up information that isn't in the context.

Context from documents:
{context}

User Question: {question}

Answer (based on the context above):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

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

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        print(f"Processing {len(documents)} documents…")

        all_chunks = self.document_processor.process_documents(documents, metadata)
        kb_id = self.config.KB_ID or None

        if metadata:
            for i, doc_metadata in enumerate(metadata):
                doc_name = doc_metadata.get("name", f"document_{i}")
                doc_type = doc_metadata.get("type", "text")
                file_size = doc_metadata.get("file_size", 0)
                file_path = doc_metadata.get("file_path")

                doc_chunks = [chunk for chunk in all_chunks
                              if chunk.metadata.get("doc_index") == i]

                self.logger.log_document(doc_name, doc_type, len(doc_chunks),
                                         file_size, file_path, kb_id=kb_id)

        if self.vectorstore:
            self.vectorstore.add_documents(all_chunks)
            print(f"✓ Added {len(all_chunks)} chunks to vector store")

        if self.bm25:
            self.bm25.add_documents(all_chunks)
            print(f"✓ Added {len(all_chunks)} chunks to BM25 index")

        if self.graph:
            self.graph.add_chunks(all_chunks)
            print(f"✓ Added {len(all_chunks)} Chunk nodes to Neo4j")
            if self.graph_extractor:
                print(f"  Extracting entities for {len(all_chunks)} chunks (this can take a minute)…")
                self.graph_extractor.extract_and_store(all_chunks, self.graph)
                print(f"✓ Entity extraction complete")

    def add_file_to_stores(self, file_path: str, metadata: Optional[Dict] = None) -> List:
        """Fast path: vector + BM25 + MySQL log. Returns chunks for deferred graph work."""
        chunks = self.document_processor.process_file(file_path, metadata)
        p = Path(file_path)
        doc_name = (metadata or {}).get("name", p.name)
        doc_type = (metadata or {}).get("type", p.suffix[1:] if p.suffix else "text")
        file_size = (metadata or {}).get("file_size", p.stat().st_size if p.exists() else 0)
        kb_id = self.config.KB_ID or None

        self.logger.log_document(doc_name, doc_type, len(chunks), file_size, str(p), kb_id=kb_id)

        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        if self.bm25:
            self.bm25.add_documents(chunks)

        print(f"✓ Indexed {len(chunks)} chunks from {doc_name}")
        return chunks

    def add_chunks_to_graph(self, chunks: List) -> None:
        """Slow path: Neo4j nodes + entity extraction. Safe to call from a background thread."""
        if not self.graph:
            return
        self.graph.add_chunks(chunks)
        if self.graph_extractor:
            print(f"  Extracting entities for {len(chunks)} chunks…")
            self.graph_extractor.extract_and_store(chunks, self.graph)
            print(f"✓ Entity extraction complete")

    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        kb_id = self.config.KB_ID or None

        source_documents = self.hybrid_retriever.invoke(question)

        if not source_documents:
            execution_time = time.time() - start_time
            answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure the documents contain the information you're looking for, or try rephrasing your question."
            return {
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "execution_time": execution_time,
            }

        answer = self.qa_chain.invoke(question)
        execution_time = time.time() - start_time

        sources = [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in source_documents
        ]

        context_texts = [doc.page_content for doc in source_documents]
        self.logger.log_query(
            question, answer, context_texts,
            execution_time, len(source_documents), session_id, kb_id=kb_id,
        )

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "execution_time": execution_time,
        }

    def get_history(self, limit: int = 5, session_id: Optional[str] = None) -> List[Dict]:
        kb_id = self.config.KB_ID or None
        return self.logger.get_recent_queries(limit, session_id, kb_id=kb_id)

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        return self.logger.get_conversation_history(session_id, limit)

    def get_documents(self) -> List[Dict]:
        kb_id = self.config.KB_ID or None
        return self.logger.get_all_documents(kb_id=kb_id)

    def close(self):
        self.logger.close()
        if self.graph:
            try:
                self.graph.close()
            except Exception as e:
                print(f"Warning: error closing Neo4j: {e}")
