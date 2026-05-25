from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from docmind.app.ingestion.chunker import Chunker
from docmind.app.ingestion.parser import Parser
from docmind.app.vectorstore.qdrant_client import QdrantVectorStore
from docmind.app.core.config import settings

router = APIRouter(prefix="/api", tags=["documents"])

ROOT_DIR = Path(__file__).resolve().parents[4]
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = QdrantVectorStore()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    target_path = UPLOAD_DIR / file.filename
    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    try:
        pages = Parser(str(target_path)).parse()
        chunker = Chunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks = chunker.chunk_pages(pages, str(target_path))
        vector_store.upsert_chunks(chunks)

        return {
            "success": True,
            "file_name": file.filename,
            "uploaded_path": str(target_path),
            "chunks": len(chunks),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {exc}")
