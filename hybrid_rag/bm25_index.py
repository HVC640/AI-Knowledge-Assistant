import logging

import bm25s

from .models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """Sparse BM25 retrieval index with save/load support."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, index_path: str = "./bm25_index"):
        self.k1 = k1
        self.b = b
        self.index_path = index_path
        self.retriever = None
        self.corpus: list[str] = []
        self.chunk_map: list[Chunk] = []
        self._text_to_chunk: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        self.chunk_map = chunks
        self.corpus = [chunk.text for chunk in chunks]
        self._text_to_chunk = {chunk.text: chunk for chunk in chunks}

        corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en")
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(corpus_tokens)
        logger.info("BM25 index built ✅")

    def save(self) -> None:
        if not self.retriever:
            raise RuntimeError("Index not built yet — call build() first")
        self.retriever.save(self.index_path, corpus=self.corpus)
        logger.info(f"BM25 index saved → {self.index_path}")

    def load(self, chunks: list[Chunk]) -> None:
        self.chunk_map = chunks
        self.corpus = [chunk.text for chunk in chunks]
        self._text_to_chunk = {chunk.text: chunk for chunk in chunks}
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
        logger.info(f"BM25 index loaded ← {self.index_path}")

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if not self.retriever:
            raise RuntimeError(
                "Index not ready — call build() or load() first")

        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = self.retriever.retrieve(
            query_tokens,
            corpus=self.corpus,
            k=min(top_k, len(self.corpus)),
        )

        hits: list[RetrievedChunk] = []
        for position in range(results.shape[1]):
            text = results[0, position]
            score = float(scores[0, position])
            chunk = self._text_to_chunk.get(text)
            if not chunk:
                continue
            hits.append(RetrievedChunk(chunk=chunk, bm25_score=score))

        return hits
