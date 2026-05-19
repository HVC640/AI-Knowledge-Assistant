from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int
    chunk_index: int

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievedChunk:
    chunk: Chunk
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
