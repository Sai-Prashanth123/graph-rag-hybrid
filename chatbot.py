import uuid
from typing import Optional, List, Dict
from datetime import datetime
from rag_system import DocumentRAGSystem
from config import RAGConfig


class ChatBot:
    
    def __init__(self, rag_system: DocumentRAGSystem):
        self.rag_system = rag_system
        self.session_id = str(uuid.uuid4())
        self.conversation_history: List[Dict] = []
    
    def start_session(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        print(f"\n{'='*60}")
        print("🤖 RAG Chatbot - Session Started")
        print(f"Session ID: {self.session_id[:8]}...")
        print(f"{'='*60}\n")
        print("Commands:")
        print("  - Type your question to chat")
        print("  - '/help' - Show help menu")
        print("  - '/history' - Show conversation history")
        print("  - '/documents' - List uploaded documents")
        print("  - '/clear' - Clear conversation history")
        print("  - '/quit' or '/exit' - Exit chatbot")
        print(f"\n{'='*60}\n")
    
    def process_command(self, user_input: str) -> bool:
        user_input = user_input.strip().lower()
        
        if user_input == '/help':
            self._show_help()
            return True
        elif user_input == '/history':
            self._show_history()
            return True
        elif user_input == '/documents':
            self._show_documents()
            return True
        elif user_input == '/clear':
            self._clear_history()
            return True
        elif user_input in ['/quit', '/exit']:
            return False
        else:
            return False
    
    def _show_help(self):
        print("\n" + "="*60)
        print("HELP MENU")
        print("="*60)
        print("Commands:")
        print("  /help       - Show this help menu")
        print("  /history    - Show conversation history")
        print("  /documents  - List all uploaded documents")
        print("  /clear      - Clear current conversation history")
        print("  /quit       - Exit the chatbot")
        print("\nJust type your question to get answers from your documents!")
        print("="*60 + "\n")
    
    def _show_history(self):
        print("\n" + "="*60)
        print("CONVERSATION HISTORY")
        print("="*60)
        
        if not self.conversation_history:
            print("No conversation history yet.")
        else:
            for i, entry in enumerate(self.conversation_history, 1):
                print(f"\n[{i}] Q: {entry['query']}")
                print(f"    A: {entry['answer'][:100]}...")
                print(f"    Time: {entry.get('execution_time', 0):.2f}s")
        
        print("="*60 + "\n")
    
    def _show_documents(self):
        print("\n" + "="*60)
        print("UPLOADED DOCUMENTS")
        print("="*60)
        
        documents = self.rag_system.get_documents()
        if not documents:
            print("No documents uploaded yet.")
        else:
            for i, doc in enumerate(documents, 1):
                print(f"\n[{i}] {doc['document_name']}")
                print(f"    Type: {doc['document_type']}")
                print(f"    Chunks: {doc['num_chunks']}")
                print(f"    Size: {doc['file_size']} bytes")
                print(f"    Uploaded: {doc['upload_timestamp']}")
        
        print("="*60 + "\n")
    
    def _clear_history(self):
        self.conversation_history = []
        print("\n✓ Conversation history cleared\n")
    
    def chat(self, user_input: str) -> Optional[Dict]:
        if self.process_command(user_input):
            return None
        
        if not user_input.strip():
            print("Please enter a question or command.")
            return None
        
        print(f"\n💭 Processing your question...")
        
        try:
            response = self.rag_system.query(user_input, self.session_id)
            
            conversation_entry = {
                "query": user_input,
                "answer": response["answer"],
                "execution_time": response["execution_time"],
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(conversation_entry)
            
            print(f"\n{'='*60}")
            print("🤖 ASSISTANT:")
            print(f"{'='*60}")
            print(response["answer"])
            print(f"\n📚 Sources: {response['num_sources']} documents")
            print(f"⏱️  Time: {response['execution_time']:.2f}s")
            
            if response['num_sources'] > 0:
                print(f"\n📄 Source Documents:")
                for i, source in enumerate(response['sources'][:3], 1):
                    doc_name = source['metadata'].get('name', 'Unknown')
                    print(f"  {i}. {doc_name}")
                    print(f"     {source['content'][:100]}...")
            
            print(f"{'='*60}\n")
            
            return response
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            return None
    
    def run_interactive(self):
        self.start_session()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit']:
                    print("\n👋 Goodbye! Thanks for using RAG Chatbot.\n")
                    break
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using RAG Chatbot.\n")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}\n")


def main():
    config = RAGConfig()
    rag_system = DocumentRAGSystem(config)
    chatbot = ChatBot(rag_system)
    chatbot.run_interactive()
    rag_system.close()
