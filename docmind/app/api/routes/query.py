from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from docmind.app.vectorstore.qdrant_client import QdrantVectorStore

router = APIRouter(prefix="/api", tags=["query"])
vector_store = QdrantVectorStore()


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_document(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = vector_store.search(request.question)
    return {
        "query": request.question,
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
