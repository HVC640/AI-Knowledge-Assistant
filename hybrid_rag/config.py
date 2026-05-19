from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Retrieval
    fetch_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    rerank_threshold: Optional[float] = None

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama3-8b-8192"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "hybrid_rag"

    # BM25 index persistence
    bm25_index_path: str = "./bm25_index"

    # LLM
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
