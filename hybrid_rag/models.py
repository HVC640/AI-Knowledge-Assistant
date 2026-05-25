from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str  # unique identifier for the chunk
    text: str  # the actual text content of the chunk
    source: str  # e.g. PDF filename or document ID
    page: int  # the page number where the chunk is located
    chunk_index: int  # the index of the chunk within the document
    char_start: Optional[int] = (
        None  # optional character offset where the chunk starts in the original page text
    )
    char_end: Optional[int] = (
        None  # optional character offset where the chunk ends in the original page text
    )
    metadata: dict = field(
        default_factory=dict
    )  # additional metadata like source pdf path, offsets, etc.

    def to_payload(self) -> dict:
        payload = {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }
        payload.update(self.metadata or {})
        return payload


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
