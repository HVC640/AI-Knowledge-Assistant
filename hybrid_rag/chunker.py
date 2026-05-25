import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Chunk

logger = logging.getLogger(__name__)


class Chunker:
    """Splits page text into overlapping word-level chunks with cleaning."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        source: str,
        source_pdf: Optional[str] = None,
        start_id: int = 0,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_id = start_id

        for page_number, text in pages:
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            words = cleaned.split()
            step = max(1, self.chunk_size - self.chunk_overlap)

            for start in range(0, len(words), step):
                chunk_words = words[start: start + self.chunk_size]
                if len(chunk_words) < 20:
                    continue

                metadata = {}
                if source_pdf:
                    metadata["source_pdf"] = str(Path(source_pdf).resolve())

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=" ".join(chunk_words),
                        source=source,
                        page=page_number,
                        chunk_index=start,
                        metadata=metadata,
                    )
                )
                chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"-\n", "", text)
        return text.strip()
