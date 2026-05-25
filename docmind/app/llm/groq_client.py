import logging
from typing import List

from groq import Groq

from shared.schemas.models import Chunk
from docmind.app.core.config import settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Wraps Groq chat completions for grounded answer generation."""

    SYSTEM_PROMPT = (
        "You are a document assistant. "
        "Answer only from provided context. "
        "If answer is not found say "
        "'I could not find that in the document.'"
    )

    def __init__(
        self,
        api_key: str = settings.GROQ_API_KEY,
        model: str = settings.GROQ_MODEL_NAME,
        temperature: float = settings.GROQ_TEMPERATURE,
        max_tokens: int = settings.GROQ_MAX_TOKENS,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, query: str, chunks: List[Chunk]) -> str:
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

        if not message.content:
            return "I could not find that in the document."

        return message.content.strip()

    @staticmethod
    def _build_prompt(query: str, chunks: List[Chunk]) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            meta = f"Source_path: {chunk.source_path} | Page_num: {chunk.page_num} | Chunk_index: {chunk.chunk_index}"
            score = (
                f"Rerank Score: {chunk.rerank_score:.3f}"
                if chunk.rerank_score is not None
                else ""
            )
            blocks.append(f"[Chunk {index} | {meta} | {score}]\n{chunk.text}")

        context = "\n\n---\n\n".join(blocks)
        return f"CONTEXT:\n{context}\n\nQUERY:\n{query}\n\nANSWER:"
