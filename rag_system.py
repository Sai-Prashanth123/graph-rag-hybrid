from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import chromadb
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import RAGConfig
from logger import MySQLLogger
from document_processor import DocumentProcessor


class DocumentRAGSystem:
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self.logger = MySQLLogger(config)
        self.document_processor = DocumentProcessor(config)
        
        self._initialize_components()
    
    def _initialize_components(self):
        print("Initializing RAG System...")
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.config.AZURE_ENDPOINT,
            api_key=self.config.AZURE_API_KEY,
            api_version=self.config.AZURE_API_VERSION,
            azure_deployment=self.config.EMBEDDING_DEPLOYMENT
        )
        print("✓ Embeddings initialized")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.AZURE_ENDPOINT,
            api_key=self.config.AZURE_API_KEY,
            api_version=self.config.AZURE_API_VERSION,
            azure_deployment=self.config.DEPLOYMENT_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        print("✓ LLM initialized")
        
        chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIR
        )
        
        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name=self.config.COLLECTION_NAME,
            embedding_function=self.embeddings
        )
        print("✓ Vector store initialized")
        
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
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.TOP_K}
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ QA chain initialized")
        print("=" * 50)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        print(f"Processing {len(documents)} documents...")
        
        all_chunks = self.document_processor.process_documents(documents, metadata)
        
        if metadata:
            for i, doc_metadata in enumerate(metadata):
                doc_name = doc_metadata.get("name", f"document_{i}")
                doc_type = doc_metadata.get("type", "text")
                file_size = doc_metadata.get("file_size", 0)
                file_path = doc_metadata.get("file_path")
                
                doc_chunks = [chunk for chunk in all_chunks 
                            if chunk.metadata.get("doc_index") == i]
                
                self.logger.log_document(doc_name, doc_type, len(doc_chunks), 
                                       file_size, file_path)
        
        self.vectorstore.add_documents(all_chunks)
        print(f"✓ Added {len(all_chunks)} chunks to vector store")
    
    def add_file(self, file_path: str, metadata: Optional[Dict] = None):
        chunks = self.document_processor.process_file(file_path, metadata)
        
        if metadata:
            doc_name = metadata.get("name", Path(file_path).name)
            doc_type = metadata.get("type", "text")
            file_size = metadata.get("file_size", 0)
        else:
            file_path_obj = Path(file_path)
            doc_name = file_path_obj.name
            doc_type = file_path_obj.suffix[1:] if file_path_obj.suffix else "text"
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        
        self.logger.log_document(doc_name, doc_type, len(chunks), file_size, file_path)
        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} chunks from {doc_name} to vector store")
    
    def add_directory(self, directory_path: str, extensions: Optional[List[str]] = None):
        chunks = self.document_processor.process_directory(directory_path, extensions)
        
        doc_indices = set(chunk.metadata.get("doc_index") for chunk in chunks)
        for doc_idx in doc_indices:
            doc_chunks = [chunk for chunk in chunks if chunk.metadata.get("doc_index") == doc_idx]
            if doc_chunks:
                metadata = doc_chunks[0].metadata
                self.logger.log_document(
                    metadata.get("name", f"document_{doc_idx}"),
                    metadata.get("type", "text"),
                    len(doc_chunks),
                    metadata.get("file_size", 0),
                    metadata.get("file_path")
                )
        
        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} chunks from directory to vector store")
    
    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        source_documents = self.retriever.invoke(question)
        
        if not source_documents or len(source_documents) == 0:
            execution_time = time.time() - start_time
            answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure the documents contain the information you're looking for, or try rephrasing your question."
            
            response = {
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "execution_time": execution_time
            }
            return response
        
        answer = self.qa_chain.invoke(question)
        
        execution_time = time.time() - start_time
        
        sources = []
        for doc in source_documents:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        context_texts = [doc.page_content for doc in source_documents]
        self.logger.log_query(
            question, answer, context_texts, 
            execution_time, len(source_documents), session_id
        )
        
        response = {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "execution_time": execution_time
        }
        
        return response
    
    def get_history(self, limit: int = 5, session_id: Optional[str] = None) -> List[Dict]:
        return self.logger.get_recent_queries(limit, session_id)
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        return self.logger.get_conversation_history(session_id, limit)
    
    def get_documents(self) -> List[Dict]:
        return self.logger.get_all_documents()
    
    def close(self):
        self.logger.close()
