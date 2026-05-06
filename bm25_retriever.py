import os
import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import RAGConfig


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_if_exists()

    def add_documents(self, docs: List[Document]) -> None:
        self.documents.extend(docs)
        tokenized = [_tokenize(d.page_content) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        self.save()

    def search(self, query: str, k: int = 10) -> List[Document]:
        if self.bm25 is None or not self.documents:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(scores, self.documents), key=lambda pair: pair[0], reverse=True
        )
        return [doc for _, doc in ranked[:k]]

    def save(self) -> None:
        path = Path(self.config.BM25_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)

    def load(self) -> None:
        path = Path(self.config.BM25_INDEX_PATH)
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        if self.documents:
            tokenized = [_tokenize(d.page_content) for d in self.documents]
            self.bm25 = BM25Okapi(tokenized)

    def _load_if_exists(self) -> None:
        try:
            self.load()
        except Exception as e:
            print(f"Warning: BM25 load failed ({e}); starting empty.")
            self.documents = []
            self.bm25 = None
