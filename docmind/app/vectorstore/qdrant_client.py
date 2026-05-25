import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Prefetch,
    PointStruct,
    VectorParams,    
)

from docmind.app.embeddings.embedder import Embedder
from shared.schemas.models import Chunk
from docmind.app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = settings.COLLECTION_NAME,
        qdrant_url: str = settings.QDRANT_URL,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_model = Embedder(model_name=embedding_model_name)


    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_model.dimension, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[Chunk]):
        self.create_collection()

        points = []
        embeddings = self.embedding_model.embed_chunks(chunks)  # Precompute embeddings for all chunks
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "page_num": chunk.page_num,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata,
                }
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)


    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Chunk]:
        if not self.client.collection_exists(self.collection_name):
            return []

        query_embedding = self.embedding_model.embed_text(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        ).points
        retrieved_chunks = []
        for result in search_result:
            retrieved_chunk = Chunk(
                id=result.id,
                chunk_id=result.payload.get("chunk_id", ""),
                source_path=result.payload.get("source_path", ""),
                page_num=result.payload.get("page_num", 0),
                chunk_index=result.payload.get("chunk_index", 0),
                text=result.payload["text"],
                metadata=result.payload,
                score=result.score,
            )
            retrieved_chunks.append(retrieved_chunk)
        return retrieved_chunks

