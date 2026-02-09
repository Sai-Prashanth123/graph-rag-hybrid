# RAG Chatbot System

A modular Retrieval-Augmented Generation (RAG) chatbot system that uses Azure OpenAI, ChromaDB, and MySQL for document-based question answering.

## Features

- 🤖 **Interactive Chatbot Interface** - Chat with your documents in real-time
- 📚 **Document Management** - Upload and process documents from files or directories
- 💾 **MySQL Integration** - Store query history, document metadata, and conversation sessions
- 🔍 **Vector Search** - Fast semantic search using ChromaDB
- 🧩 **Modular Architecture** - Clean, maintainable code structure

## Project Structure

```
.
├── config.py              # Configuration settings
├── logger.py              # MySQL logging and database operations
├── document_processor.py  # Document loading and processing
├── rag_system.py          # Core RAG system implementation
├── chatbot.py             # Interactive chatbot interface
├── main.py                # Main entry point
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MySQL

Update MySQL settings in `config.py`:

```python
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "your_password"
MYSQL_DATABASE = "rag_system"
```

### 3. Configure Azure OpenAI

Update Azure OpenAI settings in `config.py`:

```python
AZURE_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_API_KEY = "your-api-key"
AZURE_API_VERSION = "2023-05-15"
DEPLOYMENT_NAME = "gpt-4o"
EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
```

Or set environment variables:
```bash
export AZURE_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_API_KEY="your-api-key"
export MYSQL_PASSWORD="your_password"
```

## Usage

### Run the Chatbot

```bash
python main.py
```

### Adding Documents

When you run the chatbot, you'll be prompted to add documents:

1. **Add documents from a directory** - Process all files in a folder
2. **Add a single file** - Process one specific file
3. **Add text directly** - Paste text directly into the system
4. **Skip** - Use existing documents in the vector store

### Chatbot Commands

- Type your question to get answers from your documents
- `/help` - Show help menu
- `/history` - Show conversation history
- `/documents` - List uploaded documents
- `/clear` - Clear conversation history
- `/quit` or `/exit` - Exit chatbot

## Modules

### config.py
Centralized configuration management with environment variable support.

### logger.py
MySQL database operations:
- Query history logging
- Document metadata tracking
- Conversation session management

### document_processor.py
Document processing utilities:
- File loading (text files)
- Directory processing
- Text chunking with metadata

### rag_system.py
Core RAG functionality:
- Vector store management (ChromaDB)
- Document retrieval
- Question answering with context

### chatbot.py
Interactive chatbot interface:
- Command processing
- Conversation management
- User interaction handling

## Database Schema

The system automatically creates the following MySQL tables:

- `query_history` - Stores all queries and responses
- `document_metadata` - Tracks uploaded documents
- `conversation_sessions` - Manages conversation sessions

## Example Usage

```python
from config import RAGConfig
from rag_system import DocumentRAGSystem

# Initialize
config = RAGConfig()
rag_system = DocumentRAGSystem(config)

# Add documents
rag_system.add_file("document.txt")
# or
rag_system.add_directory("./documents", extensions=['.txt', '.md'])

# Query
response = rag_system.query("What is this document about?")
print(response["answer"])

# Cleanup
rag_system.close()
```

## Requirements

- Python 3.8+
- MySQL 5.7+ or MySQL 8.0+
- Azure OpenAI account
- Required Python packages (see requirements.txt)

## Notes

- Documents are stored in `./chroma_db` directory
- MySQL database is created automatically if it doesn't exist
- All queries and responses are logged to MySQL for history tracking

