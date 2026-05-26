from typing import List

from shared.schemas.models import Chunk


def dedupe_chunks(
    chunks: List[Chunk]
) -> List[Chunk]:
    seen = set()
    result = []

    for chunk in chunks:
        if chunk.id in seen:
            continue

        seen.add(chunk.id)
        result.append(chunk)

    return result