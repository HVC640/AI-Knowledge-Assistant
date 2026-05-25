from sentence_transformers import SentenceTransformer
from shared.schemas.models import Chunk
from docmind.app.core.config import settings

import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        logger.debug(f"Embedding text length={len(text)}")
        return self.model.encode(text, show_progress_bar=True).tolist()

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = settings.EMBED_CHUNK_SIZE) -> list[list[float]]:
        logger.debug(
            f"Embedding {len(chunks)} chunks"
        )  # Log the number of chunks being embedded
        return self.model.encode(
            [chunk.text for chunk in chunks], batch_size=batch_size, show_progress_bar=True
        ).tolist()
