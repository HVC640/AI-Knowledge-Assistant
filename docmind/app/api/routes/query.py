from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from docmind.app.vectorstore.qdrant_client import QdrantVectorStore
from docmind.app.retrieval.reranker import Reranker
from docmind.app.llm.groq_client import GroqClient
from docmind.app.retrieval.utils import dedupe_chunks

router = APIRouter(prefix="/api", tags=["query"])
vector_store = QdrantVectorStore()
groq_client = GroqClient()
reranker = Reranker()

class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_document(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Multi-query
    queries = groq_client.generate_queries(request.question)
    all_results = []
    for q in queries:
        results = vector_store.search(q)
        all_results.extend(results)

    # HyDE
    hypothetical_answer = groq_client.generate_hypothetical_answer(
        request.question
    )
    hyde_results = vector_store.search(
        hypothetical_answer
    )
    all_results.extend(
        hyde_results
    )

    # dedupe results based on chunk_id and source_path
    unique_results = dedupe_chunks(all_results)
    results = reranker.rerank(request.question, unique_results)
    answer = groq_client.answer(request.question, results)

    return {
        "query": request.question,
        "answer": answer,
        "results": [
            {
                "id": item.id,
                "chunk_id": item.chunk_id,
                "score": item.score,
                "text": item.text,
                "source_path": item.source_path,
                "page_num": item.page_num,
                "chunk_index": item.chunk_index,
                "metadata": item.metadata,
            }
            for item in results
        ],
    }
