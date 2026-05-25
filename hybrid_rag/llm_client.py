import logging
from typing import List

from groq import Groq

from .models import RetrievedChunk

logger = logging.getLogger(__name__)


class LLMClient:
    """Wraps Groq chat completions for grounded answer generation."""

    SYSTEM_PROMPT = (
        "You are a precise assistant. Answer questions using ONLY the provided context."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        prompt = self._build_prompt(query, chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        message = response.choices[0].message
        return message.content.strip()

    @staticmethod
    def _build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for index, retrieved in enumerate(chunks, start=1):
            meta = f"Source: {retrieved.chunk.source} | Page: {retrieved.chunk.page}"
            score = (
                f"Rerank Score: {retrieved.rerank_score:.3f}"
                if retrieved.rerank_score is not None
                else ""
            )
            blocks.append(f"[Chunk {index} | {meta} | {score}]\n{retrieved.chunk.text}")

        context = "\n\n---\n\n".join(blocks)
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
