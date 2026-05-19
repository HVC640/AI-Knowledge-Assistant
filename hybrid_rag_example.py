import logging
from hybrid_rag import HybridRAG, RAGConfig
import os
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)


def main() -> None:
    config = RAGConfig(
        chunk_size=300,
        chunk_overlap=50,
        fetch_k=20,
        rrf_k=60,
        rerank_top_k=10,
        rerank_threshold=None,
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model="openai/gpt-oss-120b",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="hybrid_rag",
        bm25_index_path="./bm25_index",
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    pdf_path = os.environ.get("HYBRID_RAG_PDF")

    rag = HybridRAG(config=config, groq_api_key=groq_api_key)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rag.ingest(pdf_path)

    result = rag.query(
        "Tell me about CONDITIONS PRECEDENT TO CREDIT EXTENSIONS", verbose=True)
    print("\nAnswer:\n", result.answer)


if __name__ == "__main__":
    main()
