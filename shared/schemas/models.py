from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str
    chunk_id: str
    text: str
    source_path: str
    page_num: int
    chunk_index: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    score: Optional[float] = None

    def to_payload(self) -> dict:
        payload = {
            "text": self.text,
            "source_path": self.source_path,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
        }
        payload.update(self.metadata or {})
        return payload
