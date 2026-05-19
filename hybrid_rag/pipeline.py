import logging
from pathlib import Path
from typing import List

from .config import RAGConfig
from .bm25_index import BM25Index
from .chunker import Chunker
from .fusion import RRFFusion
from .llm_client import LLMClient
from .models import RAGResult, RetrievedChunk
from .pdf_loader import PDFLoader
from .reranker import Reranker
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRAG:
    """End-to-end hybrid retrieval augmented generation pipeline."""

    def __init__(self, config: RAGConfig, groq_api_key: str):
        self.config = config
        self.loader = None
        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self.bm25 = BM25Index(index_path=config.bm25_index_path)
        self.vector = VectorStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
        )
        self.fusion = RRFFusion(k=config.rrf_k)
        self.reranker = Reranker(model=config.reranker_model)
        self.llm = LLMClient(
            api_key=groq_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self._chunks: List = []
        logger.info("HybridRAG pipeline initialized ✅")

    def ingest(self, pdf_path: str) -> None:
        logger.info(f"▶ Ingesting: {pdf_path}")
        self.loader = PDFLoader(pdf_path)
        pages = self.loader.load()
        source_name = Path(pdf_path).name
        self._chunks = self.chunker.chunk_pages(pages, source=source_name)

        if not self._chunks:
            raise ValueError(
                "No chunks produced — check if PDF has extractable text")

        self.bm25.build(self._chunks)
        self.bm25.save()
        self.vector.build(self._chunks)
        logger.info(f"✅ Ingestion complete — {len(self._chunks)} chunks ready")

    def load_existing_index(self, pdf_path: str) -> None:
        logger.info("Loading existing indexes...")
        self.loader = PDFLoader(pdf_path)
        pages = self.loader.load()
        source_name = Path(pdf_path).name
        self._chunks = self.chunker.chunk_pages(pages, source=source_name)
        self.bm25.load(self._chunks)
        logger.info("✅ Existing indexes loaded")

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        bm25_hits = self.bm25.search(query, top_k=self.config.fetch_k)
        dense_hits = self.vector.search(query, top_k=self.config.fetch_k)

        fused = self.fusion.fuse(
            bm25_hits, dense_hits, top_k=self.config.fetch_k)
        final = self.reranker.rerank(
            query,
            fused,
            top_k=self.config.rerank_top_k,
            threshold=self.config.rerank_threshold,
        )

        return final

    def query(self, question: str, verbose: bool = False) -> RAGResult:
        logger.info(f"Query: {question}")
        candidates = self.retrieve(question)

        if verbose:
            self._print_retrieval_debug(question, candidates)

        if not candidates:
            return RAGResult(
                query=question,
                answer="No relevant context found in the document.",
                retrieved_chunks=[],
            )

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
                f"BM25={retrieved.bm25_score or 0:.3f} | "
                f"Dense={retrieved.dense_score or 0:.3f} | "
                f"RRF={retrieved.rrf_score or 0:.5f} | "
                f"Rerank={retrieved.rerank_score or 0.0:.3f}"
            )
            print(f"       {retrieved.chunk.text[:120]}...")
        print()
