import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Dense vector retrieval backed by Qdrant."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "hybrid_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.dim = self.encoder.get_embedding_dimension()
        logger.info(f"Embedding dim: {self.dim}")

    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        logger.info(
            f"Embedding {len(chunks)} chunks (batch_size={batch_size})...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=chunk.id,
                vector=embeddings[index].tolist(),
                payload=chunk.to_payload(),
            )
            for index, chunk in enumerate(chunks)
        ]

        batch_size_upload = 256
        for start in range(0, len(points), batch_size_upload):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start: start + batch_size_upload],
            )

        logger.info(f"Qdrant collection '{self.collection_name}' indexed ✅")

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        query_vector = self.encoder.encode(
            query, convert_to_numpy=True).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        hits: List[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunk = Chunk(
                id=point.id,
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                page=payload.get("page", 0),
                chunk_index=payload.get("chunk_index", 0),
            )
            hits.append(RetrievedChunk(chunk=chunk, dense_score=point.score))

        return hits
