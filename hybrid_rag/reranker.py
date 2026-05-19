import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from .models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for final candidate scoring."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading re-ranker: {model}")
        self.model = CrossEncoder(model, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:        
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(
            candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)
        filtered = [
            item for item in ranked if item.rerank_score is not None and (threshold is None or item.rerank_score > threshold)]        
        return filtered[:top_k]
