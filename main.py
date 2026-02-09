import os
import sys
from pathlib import Path
from config import RAGConfig
from rag_system import DocumentRAGSystem
from chatbot import ChatBot


def setup_documents(rag_system: DocumentRAGSystem):
    print("\n" + "="*60)
    print("DOCUMENT SETUP")
    print("="*60)
    print("\nOptions:")
    print("1. Add documents from a directory")
    print("2. Add a single file")
    print("3. Add text directly")
    print("4. Skip (use existing documents)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        directory_path = input("Enter directory path: ").strip()
        path_obj = Path(directory_path)
        
        if not path_obj.exists():
            print("❌ Path not found!")
        elif path_obj.is_file():
            print(f"❌ Error: The path '{directory_path}' is a file, not a directory!")
            print(f"💡 Tip: If you want to add a single file, please use option 2 (Add a single file)")
        elif path_obj.is_dir():
            extensions_input = input("Enter file extensions (comma-separated, e.g., txt,md,pdf) or press Enter for default: ").strip()
            extensions = [f".{ext.strip()}" for ext in extensions_input.split(",")] if extensions_input else None
            try:
                rag_system.add_directory(directory_path, extensions)
                print("✓ Documents added successfully!")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Tip: Make sure the directory contains files with the specified extensions")
        else:
            print("❌ Invalid path!")
    
    elif choice == "2":
        file_path = input("Enter file path: ").strip()
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            print("❌ File not found!")
        elif path_obj.is_dir():
            print(f"❌ Error: The path '{file_path}' is a directory, not a file!")
            print(f"💡 Tip: If you want to add multiple files from a directory, please use option 1 (Add documents from a directory)")
        elif path_obj.is_file():
            try:
                rag_system.add_file(file_path)
                print("✓ File added successfully!")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Tip: Make sure the file is readable and in a supported format")
        else:
            print("❌ Invalid path!")
    
    elif choice == "3":
        print("Enter your text (press Enter twice to finish):")
        lines = []
        empty_count = 0
        while True:
            line = input()
            if not line:
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        
        text = "\n".join(lines)
        if text.strip():
            doc_name = input("Enter document name (optional): ").strip() or "user_input.txt"
            metadata = [{"name": doc_name, "type": "text"}]
            rag_system.add_documents([text], metadata)
            print("✓ Text added successfully!")
        else:
            print("❌ No text provided!")
    
    elif choice == "4":
        print("Skipping document setup...")
    
    elif choice == "5":
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice!")


def main():
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT SYSTEM")
    print("="*60)
    print("\nInitializing system...")
    
    config = RAGConfig()
    
    try:
        rag_system = DocumentRAGSystem(config)
    except Exception as e:
        print(f"\n❌ Error initializing RAG system: {e}")
        print("\nPlease check:")
        print("  - MySQL connection settings in config.py")
        print("  - Azure OpenAI credentials")
        print("  - Network connectivity")
        sys.exit(1)
    
    try:
        setup_documents(rag_system)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        rag_system.close()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during document setup: {e}")
    
    try:
        chatbot = ChatBot(rag_system)
        chatbot.run_interactive()
    except Exception as e:
        print(f"\n❌ Error running chatbot: {e}")
    finally:
        rag_system.close()
        print("✓ System closed successfully")


if __name__ == "__main__":
    main()
