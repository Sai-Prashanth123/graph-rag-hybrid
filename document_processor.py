import os
from typing import List, Dict, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import RAGConfig

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
except ImportError:
    PyPDFLoader = None
    TextLoader = None
    UnstructuredMarkdownLoader = None


class DocumentProcessor:
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_file(self, file_path: str) -> str:
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        if file_ext == '.pdf':
            if PyPDFLoader is None:
                raise Exception("PDF support requires 'pypdf' package. Install it with: pip install pypdf")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                text = "\n\n".join([doc.page_content for doc in documents])
                return text
            except Exception as e:
                raise Exception(f"Error loading PDF file {file_path}: {e}")
        
        elif file_ext in ['.md', '.markdown']:
            if UnstructuredMarkdownLoader is not None:
                try:
                    loader = UnstructuredMarkdownLoader(file_path)
                    documents = loader.load()
                    return "\n\n".join([doc.page_content for doc in documents])
                except:
                    pass
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise Exception(f"Error loading file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {e}")
    
    def load_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> List[Dict]:
        if extensions is None:
            extensions = ['.txt', '.md', '.pdf', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml']
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise Exception(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise Exception(f"Path is not a directory: {directory_path}")
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    text = self.load_file(str(file_path))
                    metadata = {
                        "name": file_path.name,
                        "type": file_path.suffix[1:] if file_path.suffix else "unknown",
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size
                    }
                    documents.append({
                        "text": text,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        if len(documents) == 0:
            ext_list = ', '.join(extensions)
            raise Exception(f"No files found with extensions [{ext_list}] in directory: {directory_path}")
        
        return documents
    
    def process_documents(self, documents: List[str],
                         metadata: Optional[List[Dict]] = None) -> List[Document]:
        all_chunks = []

        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)

            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            doc_metadata["doc_index"] = i

            doc_name = doc_metadata.get("name", f"document_{i}")

            for j, chunk in enumerate(chunks):
                chunk_metadata = doc_metadata.copy()
                chunk_metadata["chunk_index"] = j
                chunk_metadata["chunk_id"] = f"{doc_name}::{j}"
                all_chunks.append(Document(page_content=chunk, metadata=chunk_metadata))

        return all_chunks
    
    def process_file(self, file_path: str, metadata: Optional[Dict] = None) -> List[Document]:
        text = self.load_file(file_path)
        
        if metadata is None:
            file_path_obj = Path(file_path)
            metadata = {
                "name": file_path_obj.name,
                "type": file_path_obj.suffix[1:] if file_path_obj.suffix else "text",
                "file_path": file_path,
                "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else 0
            }
        
        return self.process_documents([text], [metadata])
    
    def process_directory(self, directory_path: str, 
                         extensions: Optional[List[str]] = None) -> List[Document]:
        file_docs = self.load_directory(directory_path, extensions)
        all_chunks = []
        
        for file_doc in file_docs:
            chunks = self.process_documents([file_doc["text"]], [file_doc["metadata"]])
            all_chunks.extend(chunks)
        
        return all_chunks
