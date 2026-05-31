from docmind.app.ingestion.parser import Parser
from docmind.app.ingestion.chunker import Chunker
from docmind.app.retrieval.utils import dedupe_chunks
from docmind.app.vectorstore.qdrant_client import QdrantVectorStore
from docmind.app.retrieval.reranker import Reranker
from docmind.app.llm.groq_client import GroqClient

if __name__ == "__main__":
    # source_path = "C:/Workspace/projects/AI-Knowledge-Assistant/docs/Stryker Corporation.pdf"
    query = "What are the CONDITIONS PRECEDENT TO CREDIT EXTENSIONS ?"
    groq_client = GroqClient()
    vector_store = QdrantVectorStore()
    
    # pages = Parser(source_path).parse()
    # chunker = Chunker()
    # chunks = chunker.chunk_pages(pages, source_path)
    # print(f"Total chunks created: {len(chunks)}")
    # vector_store.upsert_chunks(chunks)

    # Multi-query
    queries = groq_client.generate_queries(query)
    print(f"\nGenerated queries: {queries}\n")
    all_results = []
    for q in queries:
        results = vector_store.search(q)
        all_results.extend(results)

    # HyDE
    hypothetical_answer = groq_client.generate_hypothetical_answer(
        query
    )
    print(f"\nHypothetical answer: {hypothetical_answer}\n")
    hyde_results = vector_store.search(
        hypothetical_answer
    )
    all_results.extend(
        hyde_results
    )

    # dedupe results based on chunk_id and source_path
    unique_results = dedupe_chunks(all_results)
    print(f"\nTotal unique chunks retrieved: {len(unique_results)}\n")
    
    reranker = Reranker()
    results = reranker.rerank(query, unique_results)

    answer = groq_client.answer(query, results)
    print(f"\nAnswer:\n{answer}\n")

    print(f"\nTop {len(results)} results for query: '{query}'")
    for idx, result in enumerate(results):
        print(f"\nResult {idx + 1}:")
        print(f"ID: {result.id}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Score: {result.score}")
        print(f"Rerank Score: {result.rerank_score}")
        print(f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")

