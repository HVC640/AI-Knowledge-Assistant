import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FusionQuery,
    Prefetch,
    PointStruct,
    SparseVector,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from docmind.app.embeddings.embedder import Embedder
from docmind.app.embeddings.sparse_embedder import SparseEmbedder
from shared.schemas.models import Chunk
from docmind.app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = settings.COLLECTION_NAME,
        qdrant_url: str = settings.QDRANT_URL,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME,
        sparse_embedding_model_name: str = settings.SPARSE_EMBEDDING_MODEL_NAME,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.dense_embedding_model = Embedder(model_name=embedding_model_name)
        self.sparse_embedder = SparseEmbedder(model_name=sparse_embedding_model_name)
        self.dense_name = settings.DENSE_FIELD_NAME
        self.sparse_name = settings.SPARSE_FIELD_NAME

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_name: VectorParams(
                        size=self.dense_embedding_model.dimension, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.sparse_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )

    def upsert_chunks(self, chunks: List[Chunk]):
        self.create_collection()

        points = []
        dense_embeddings = self.dense_embedding_model.embed_chunks(chunks)
        sparse_embeddings = self.sparse_embedder.embed_chunks(chunks)

        for chunk, dense_embedding, sparse_embedding in zip(chunks, dense_embeddings, sparse_embeddings):
            point = PointStruct(
                id=chunk.id,
                vector={
                    self.dense_name: dense_embedding,
                    self.sparse_name: SparseVector(
                        indices=sparse_embedding.indices,
                        values=sparse_embedding.values,
                    )
                },
                payload={
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "page_num": chunk.page_num,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata,
                },
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Chunk]:
        if not self.client.collection_exists(self.collection_name):
            return []

        dense_embedding = self.dense_embedding_model.embed_text(query)
        sparse_embedding = self.sparse_embedder.embed_text(query)

        prefetch = [
            Prefetch(
                query=dense_embedding,
                using=self.dense_name,
                limit=top_k
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values
                ),
                using=self.sparse_name,
                limit=top_k
            )
        ]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=FusionQuery(fusion="rrf"),
            prefetch=prefetch,
            limit=top_k,
            with_payload=True,
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
