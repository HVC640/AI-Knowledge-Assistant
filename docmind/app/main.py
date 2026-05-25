from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from docmind.app.api.routes.documents import router as documents_router
from docmind.app.api.routes.query import router as query_router

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "docmind" / "app" / "static"

app = FastAPI(title="DocMind Qdrant Demo")
app.include_router(documents_router)
app.include_router(query_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
