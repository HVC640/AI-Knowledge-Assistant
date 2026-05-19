import logging
from typing import Dict, List

from .models import RetrievedChunk

logger = logging.getLogger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion between BM25 and dense retrieval results."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_hits: List[RetrievedChunk],
        dense_hits: List[RetrievedChunk],
        top_k: int = 20,
    ) -> List[RetrievedChunk]:
        rrf_scores: Dict[int, float] = {}
        registry: Dict[int, RetrievedChunk] = {}

        for rank, hit in enumerate(bm25_hits):
            chunk_id = hit.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(
                chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            registry[chunk_id] = hit

        for rank, hit in enumerate(dense_hits):
            chunk_id = hit.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(
                chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            existing = registry.get(chunk_id)
            if existing:
                existing.dense_score = hit.dense_score
            else:
                registry[chunk_id] = hit

        ranked = sorted(rrf_scores.items(),
                        key=lambda pair: pair[1], reverse=True)
        fused_hits: List[RetrievedChunk] = []
        for chunk_id, score in ranked[:top_k]:
            hit = registry[chunk_id]
            hit.rrf_score = round(score, 6)
            fused_hits.append(hit)

        return fused_hits
