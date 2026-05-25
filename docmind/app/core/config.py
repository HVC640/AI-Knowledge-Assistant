from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBED_CHUNK_SIZE: int = 32
    
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MAX_LENGTH: int = 512

    LLM_MODEL_NAME: str = "llama3-8b-8192"

    GROQ_API_KEY: str = ""

    COLLECTION_NAME: str = "documents"
    TOP_K: int = 10
    RERANKER_TOP_K: int = 3
    RERANKER_THRESHOLD: Optional[float] = None

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    class Config:
        ENV_FILE = ".env"


settings = Settings()