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

    # Qdrant vector names / sparse settings
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    sparse_index_on_disk: bool = True

    # LLM
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
