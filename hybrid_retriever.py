from collections import defaultdict
from typing import List

from langchain_core.documents import Document


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
