import logging
from pathlib import Path
from typing import List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from .models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Dense + sparse vector management in Qdrant.

    Uses a single Qdrant collection with named dense and sparse vector storages.
    All vector data and metadata is persisted in Qdrant; no local chunk index
    files are required for retrieval.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "hybrid_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
        sparse_dim: int = 4096,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        sparse_index_on_disk: bool = True,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.dim = self.encoder.get_embedding_dimension()
        self.sparse_dim = sparse_dim
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.sparse_index_on_disk = sparse_index_on_disk
        self._sparse_vectorizer: Optional[TfidfVectorizer] = None
        logger.info(
            f"Embedding dim: {self.dim} | sparse_dim: {self.sparse_dim}")

    def set_collection(self, collection_name: str) -> None:
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def _ensure_collection(self) -> None:
        if not self.collection_exists():
            vectors_conf = {
                self.dense_vector_name: VectorParams(
                    size=self.dim, distance=Distance.COSINE)
            }
            sparse_conf = {
                self.sparse_vector_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=self.sparse_index_on_disk)
                )
            }
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_conf,
                sparse_vectors_config=sparse_conf,
            )

    def _build_source_filter(self, source_filter: Optional[Union[str, List[str]]]) -> Optional[Filter]:
        if source_filter is None:
            return None

        if isinstance(source_filter, str):
            values = [str(Path(source_filter).resolve())]
        else:
            values = [str(Path(item).resolve()) for item in source_filter]

        if len(values) == 1:
            return Filter(
                must=[
                    FieldCondition(
                        key="source_pdf",
                        match=MatchValue(value=values[0]),
                    )
                ]
            )

        return Filter(
            must=[
                FieldCondition(
                    key="source_pdf",
                    match=MatchAny(any=values),
                )
            ]
        )

    def _scroll_chunks(self, query_filter: Optional[Filter] = None) -> List[Chunk]:
        chunks: List[Chunk] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                scroll_filter=query_filter,
                offset=offset,
                limit=100,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                chunks.append(
                    Chunk(
                        id=str(point.id),
                        text=payload.get("text", ""),
                        source=payload.get("source", ""),
                        page=payload.get("page", 0),
                        chunk_index=payload.get("chunk_index", 0),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in {"text", "source", "page", "chunk_index"}
                        },
                    )
                )

            if offset is None:
                break

        return chunks

    def has_pdf(self, pdf_path: str) -> bool:
        if not self.collection_exists():
            return False

        query_filter = self._build_source_filter(pdf_path)
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=False,
            with_vectors=False,
            scroll_filter=query_filter,
            limit=1,
        )
        return bool(points)

    def _rebuild_sparse_vectorizer(self) -> None:
        chunks = self._scroll_chunks()
        texts = [chunk.text for chunk in chunks]
        self._sparse_vectorizer = TfidfVectorizer(max_features=self.sparse_dim)
        self._sparse_vectorizer.fit(texts)

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        self._ensure_collection()

        all_chunks = self._scroll_chunks() if self.collection_exists() else []
        all_chunks.extend(chunks)

        texts = [chunk.text for chunk in all_chunks]
        dense_emb = self.encoder.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        self._sparse_vectorizer = TfidfVectorizer(max_features=self.sparse_dim)
        sparse_csr = self._sparse_vectorizer.fit_transform(texts)

        points: List[PointStruct] = []
        for index, chunk in enumerate(all_chunks):
            row = sparse_csr.getrow(index)
            sparse_vector = SparseVector(
                indices=row.indices.tolist(), values=row.data.tolist()
            )
            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector={
                        self.dense_vector_name: dense_emb[index].tolist(),
                        self.sparse_vector_name: sparse_vector,
                    },
                    payload=chunk.to_payload(),
                )
            )

        for start in range(0, len(points), batch_size):
            batch = points[start: start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name, points=batch)

    def _build_query_chunks(self, query: str) -> SparseVector:
        if self._sparse_vectorizer is None:
            self._rebuild_sparse_vectorizer()

        if self._sparse_vectorizer is None:
            raise RuntimeError("Unable to build sparse vector representation")

        qcsr = self._sparse_vectorizer.transform([query])
        row = qcsr.getrow(0)
        return SparseVector(indices=row.indices.tolist(), values=row.data.tolist())

    def _payload_to_chunk(self, point) -> Chunk:
        payload = point.payload or {}
        return Chunk(
            id=str(point.id),
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            page=payload.get("page", 0),
            chunk_index=payload.get("chunk_index", 0),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in {"text", "source", "page", "chunk_index"}
            },
        )

    def search_dense(
        self, query: str, top_k: int = 20, source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[RetrievedChunk]:
        self._ensure_collection()
        qvec = self.encoder.encode(query, convert_to_numpy=True).tolist()
        query_filter = self._build_source_filter(source_filter)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            using=self.dense_vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: List[RetrievedChunk] = []
        for point in results.points:
            hits.append(
                RetrievedChunk(
                    chunk=self._payload_to_chunk(point),
                    dense_score=point.score,
                )
            )
        return hits

    def search_sparse(
        self, query: str, top_k: int = 20, source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[RetrievedChunk]:
        self._ensure_collection()
        query_filter = self._build_source_filter(source_filter)
        sparse_query = self._build_query_chunks(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_query,
            using=self.sparse_vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: List[RetrievedChunk] = []
        for point in results.points:
            hits.append(
                RetrievedChunk(
                    chunk=self._payload_to_chunk(point),
                    sparse_score=point.score,
                )
            )
        return hits

    def search(
        self, query: str, top_k: int = 20, source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[RetrievedChunk]:
        dense = self.search_dense(
            query, top_k=top_k, source_filter=source_filter)
        sparse = self.search_sparse(
            query, top_k=top_k, source_filter=source_filter)
        return dense + sparse
