from collections import defaultdict
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable


def rrf_fuse(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009).

    Combines multiple ranked lists into one ranking using only ranks
    (not scores), since scores from different retrievers aren't comparable.
    """
    scores: dict = defaultdict(float)
    docs_by_id: dict = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = doc.metadata.get("chunk_id")
            if cid is None:
                continue
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in docs_by_id:
                docs_by_id[cid] = doc

    sorted_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])
    return [docs_by_id[cid] for cid in sorted_ids]


class HybridRetriever(Runnable):
    """
    Runs vector + BM25 + graph retrievers and fuses via RRF.
    Any retriever can be None — it will be skipped gracefully.
    """

    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: Any,
        graph_retriever: Any,
        graph_extractor: Any,
        config,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.graph_extractor = graph_extractor
        self.config = config

    def invoke(self, question: str, *args, **kwargs) -> List[Document]:
        rankings: List[List[Document]] = []

        if self.vector_retriever is not None:
            try:
                rankings.append(self.vector_retriever.invoke(question) or [])
            except Exception as e:
                print(f"Warning: vector retrieval failed: {e}")

        if self.bm25_retriever is not None:
            try:
                rankings.append(
                    self.bm25_retriever.search(question, k=self.config.RETRIEVER_K) or []
                )
            except Exception as e:
                print(f"Warning: BM25 retrieval failed: {e}")

        if self.graph_retriever is not None and self.graph_extractor is not None:
            try:
                names = self.graph_extractor._extract_entity_names(question)
                rankings.append(
                    self.graph_retriever.search(
                        names, k=self.config.RETRIEVER_K, hops=self.config.GRAPH_HOPS
                    ) or []
                )
            except Exception as e:
                print(f"Warning: graph retrieval failed: {e}")

        fused = rrf_fuse(rankings, k=self.config.RRF_K)
        return fused[: self.config.TOP_K]
