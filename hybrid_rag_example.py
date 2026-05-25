import logging
import os
from typing import List

from dotenv import load_dotenv
from hybrid_rag import HybridRAG, RAGConfig

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)


def _pdf_paths() -> List[str]:
    primary = os.environ.get("HYBRID_RAG_PDF", "./pdf/document1.pdf")
    secondary = os.environ.get("HYBRID_RAG_PDF_2", "./pdf/document2.pdf")
    return [primary, secondary]


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
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run this example")

    rag = HybridRAG(config=config, groq_api_key=groq_api_key)
    collection_name = "hybrid_rag"

    pdf_paths = [path for path in _pdf_paths(
    ) if path and os.path.exists(path)]
    if not pdf_paths:
        raise FileNotFoundError(
            "No PDF files found. Set HYBRID_RAG_PDF or HYBRID_RAG_PDF_2 to valid paths."
        )

    for pdf_path in pdf_paths:
        print(f"Indexing or loading PDF: {pdf_path}")
        rag.load_or_ingest(pdf_path, collection_name=collection_name)

    print("\nQuerying the entire collection...")
    collection_result = rag.query(
        "What is the main argument of the document collection?",
        collection_name=collection_name,
        verbose=True,
    )
    print("\nCollection answer:\n", collection_result.answer)

    if len(pdf_paths) > 1:
        print(f"\nQuerying only the first PDF: {pdf_paths[0]}")
        single_result = rag.query(
            "Summarize the key points of the first PDF.",
            collection_name=collection_name,
            pdf_path=pdf_paths[0],
            verbose=True,
        )
        print("\nSingle PDF answer:\n", single_result.answer)

        print(f"\nQuerying only the second PDF: {pdf_paths[1]}")
        second_result = rag.query(
            "Summarize the key points of the second PDF.",
            collection_name=collection_name,
            pdf_path=pdf_paths[1],
            verbose=True,
        )
        print("\nSecond PDF answer:\n", second_result.answer)

        print(
            f"\nQuerying with a multi-PDF filter: {pdf_paths[0]} and {pdf_paths[1]}")
        multi_result = rag.query(
            "What common themes appear in both documents?",
            collection_name=collection_name,
            pdf_path=pdf_paths[:2],
            verbose=True,
        )
        print("\nMulti-PDF answer:\n", multi_result.answer)


if __name__ == "__main__":
    main()
