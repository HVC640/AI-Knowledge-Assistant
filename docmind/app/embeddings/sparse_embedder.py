import logging
from fastembed import SparseTextEmbedding

from shared.schemas.models import Chunk
from docmind.app.core.config import settings

logger = logging.getLogger(__name__)


class SparseEmbedder:
    def __init__(
        self,
        model_name: str = settings.SPARSE_EMBEDDING_MODEL_NAME
    ):
        logger.info(
            f"Loading sparse embedder: {model_name}"
        )
        self.model = SparseTextEmbedding(model_name)

    def embed_text(self, text: str):
        return list(
            self.model.embed([text])
        )[0]

    def embed_chunks(self, chunks: list[Chunk]):
        logger.debug(
            f"Embedding {len(chunks)} chunks"
        )
        return list(
            self.model.embed([chunk.text for chunk in chunks])
        )