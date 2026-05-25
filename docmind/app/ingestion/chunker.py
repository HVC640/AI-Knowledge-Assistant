import logging
import re
from typing import List, Tuple
import uuid

from shared.schemas.models import Chunk

logger = logging.getLogger(__name__)


class Chunker:
    """Splits page text into overlapping word-level chunks with cleaning."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        source_path: str,
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

                chunks.append(
                    Chunk(
                        id=uuid.uuid4().hex,
                        chunk_id=f"{source_path}:{page_number}:{chunk_id}",
                        text=" ".join(chunk_words),
                        source_path=source_path,
                        page_num=page_number,
                        chunk_index=start,
                        metadata={
                            "word_count": len(chunk_words),
                            "source": source_path,
                            "page": page_number,
                        },
                    )
                )
                chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
