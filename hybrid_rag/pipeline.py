import logging
from pathlib import Path
from typing import List, Optional, Dict, Union

from .config import RAGConfig
from .chunker import Chunker
from .llm_client import LLMClient
from .models import RAGResult, RetrievedChunk
from .pdf_loader import PDFLoader
from .reranker import Reranker
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRAG:
    """End-to-end hybrid retrieval augmented generation pipeline.

    This simplified pipeline stores both dense and migrated sparse (TF-IDF)
    vectors in Qdrant and queries them for retrieval. BM25 has been
    migrated into sparse Qdrant vectors; there is no separate BM25 index.
    """

    def __init__(self, config: RAGConfig, groq_api_key: str):
        self.config = config
        self.collection_name: Optional[str] = None
        self.loader = None
        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self.vector = VectorStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            dense_vector_name=getattr(config, "dense_vector_name", "dense"),
            sparse_vector_name=getattr(config, "sparse_vector_name", "sparse"),
            sparse_index_on_disk=getattr(config, "sparse_index_on_disk", True),
        )
        self.reranker = Reranker(model=config.reranker_model)
        self.llm = LLMClient(
            api_key=groq_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        logger.info("HybridRAG pipeline initialized ✅")

    def set_collection(self, collection_name: str) -> None:
        if self.collection_name == collection_name:
            return
        self.collection_name = collection_name
        self.vector.set_collection(collection_name)

    def ingest(self, pdf_path: str, collection_name: str = "default") -> None:
        self.set_collection(collection_name)

        if self.vector.collection_exists() and self.vector.has_pdf(pdf_path):
            logger.info(
                f"PDF already indexed under collection '{collection_name}' — skipping ingestion"
            )
            return

        logger.info(
            f"▶ Ingesting PDF into collection '{collection_name}': {pdf_path}")
        self.loader = PDFLoader(pdf_path)
        pages = self.loader.load()
        source_name = Path(pdf_path).name

        new_chunks = self.chunker.chunk_pages(
            pages, source=source_name, source_pdf=pdf_path)

        if not new_chunks:
            raise ValueError(
                "No chunks produced — check if PDF has extractable text")

        self.vector.upsert_chunks(new_chunks)
        logger.info(
            f"✅ Ingestion complete — {len(new_chunks)} new chunks added to collection '{collection_name}'")

    def load_collection(self, collection_name: str) -> None:
        self.set_collection(collection_name)

        if not self.vector.collection_exists():
            raise RuntimeError(
                f"Collection '{collection_name}' does not exist or has no indexed documents")

        logger.info(f"✅ Collection '{collection_name}' loaded")

    def is_indexed(self, pdf_path: str, collection_name: str = "default") -> bool:
        self.set_collection(collection_name)
        sparse_ready = self.vector.collection_exists() and self.vector.has_pdf(pdf_path)
        collection_ready = self.vector.collection_exists()
        if not sparse_ready:
            logger.info(
                "Sparse vector index does not include this PDF in the collection")
        if not collection_ready:
            logger.info("Qdrant collection does not exist")
        return sparse_ready and collection_ready

    def load_or_ingest(self, pdf_path: str, collection_name: str = "default") -> None:
        if self.is_indexed(pdf_path, collection_name):
            logger.info("Existing index found — loading collection")
            self.load_collection(collection_name)
        else:
            self.ingest(pdf_path, collection_name)

    def retrieve(self, query: str, collection_name: str = "default", pdf_path: Optional[Union[str, List[str]]] = None) -> List[RetrievedChunk]:
        self.set_collection(collection_name)

        if not self.vector.collection_exists():
            raise RuntimeError(
                f"Collection '{collection_name}' is not indexed yet. Ingest at least one PDF")

        dense_hits = self.vector.search_dense(
            query, top_k=self.config.fetch_k, source_filter=pdf_path)
        sparse_hits = self.vector.search_sparse(
            query, top_k=self.config.fetch_k, source_filter=pdf_path)

        # Simple Reciprocal Rank Fusion
        rrf_k = getattr(self.config, "rrf_k", 60)
        rrf_scores: Dict[str, float] = {}
        registry: Dict[str, RetrievedChunk] = {}

        for rank, hit in enumerate(sparse_hits):
            cid = hit.chunk.id
            rrf_scores[cid] = rrf_scores.get(
                cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            registry[cid] = hit
            registry[cid].sparse_score = hit.sparse_score

        for rank, hit in enumerate(dense_hits):
            cid = hit.chunk.id
            rrf_scores[cid] = rrf_scores.get(
                cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid in registry:
                registry[cid].dense_score = hit.dense_score
            else:
                registry[cid] = hit

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused: List[RetrievedChunk] = []
        for cid, score in ranked[: self.config.fetch_k]:
            entry = registry[cid]
            entry.rrf_score = round(score, 6)
            fused.append(entry)

        final = self.reranker.rerank(
            query, fused, top_k=self.config.rerank_top_k, threshold=self.config.rerank_threshold)
        return final

    def query(self, question: str, collection_name: str = "default", pdf_path: Optional[Union[str, List[str]]] = None, verbose: bool = False) -> RAGResult:
        logger.info(
            f"Query: '{question}' collection='{collection_name}' pdf='{pdf_path}'")
        candidates = self.retrieve(
            question, collection_name=collection_name, pdf_path=pdf_path)

        if verbose:
            self._print_retrieval_debug(question, candidates)

        if not candidates:
            return RAGResult(query=question, answer="No relevant context found in the document.", retrieved_chunks=[])

        answer = self.llm.answer(question, candidates)
        return RAGResult(query=question, answer=answer, retrieved_chunks=candidates)

    def _print_retrieval_debug(self, query: str, chunks: List[RetrievedChunk]) -> None:
        print(f"\n{'='*65}")
        print(f"QUERY: {query}")
        print(f"{'='*65}")
        print(f"Retrieved {len(chunks)} chunks after re-ranking:\n")
        for index, retrieved in enumerate(chunks, start=1):
            print(
                f"  [{index}] Page {retrieved.chunk.page} | "
                f"Sparse={retrieved.sparse_score or 0:.3f} | "
                f"Dense={retrieved.dense_score or 0:.3f} | "
                f"RRF={retrieved.rrf_score or 0.0:.5f} | "
                f"Rerank={retrieved.rerank_score or 0.0:.3f}"
            )
            print(f"       {retrieved.chunk.text[:120]}...")
        print()
