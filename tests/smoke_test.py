"""
Manual smoke test — not run by pytest.

Usage:
    python tests/smoke_test.py

Prerequisites:
    - .env populated with Azure + Neo4j Aura credentials
    - MySQL running locally (or MYSQL_PASSWORD set in .env)
    - A PDF in the project root (defaults to Sai_Prashanth_CV.pdf)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from config import RAGConfig
from rag_system import DocumentRAGSystem


CV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Sai_Prashanth_CV.pdf")
)

QUESTIONS = [
    "What are the technical skills mentioned?",
    "Mention any acronyms or specific certifications.",
    "Who or what is connected to specific companies?",
    "Summarize the work experience.",
]


def main():
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    config = RAGConfig()
    print(f"GRAPH_ENABLED: {config.GRAPH_ENABLED}")
    print(f"BM25_ENABLED: {config.BM25_ENABLED}")
    print()

    rag = DocumentRAGSystem(config)

    if os.path.exists(CV_PATH):
        print(f"Ingesting: {CV_PATH}")
        rag.add_file(CV_PATH)
    else:
        print(f"WARNING: {CV_PATH} not found - skipping ingestion. "
              f"Will query whatever is already indexed.")

    print()
    for q in QUESTIONS:
        print("-" * 60)
        print(f"Q: {q}")
        response = rag.query(q)
        print(f"A: {response['answer'][:400]}")
        print(f"Sources: {response['num_sources']}")
        print(f"Time: {response['execution_time']:.2f}s")
        print()

    rag.close()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
