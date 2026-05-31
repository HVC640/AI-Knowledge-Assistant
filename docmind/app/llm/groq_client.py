import logging
from typing import List

from groq import Groq

from shared.schemas.models import Chunk
from docmind.app.core.config import settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Wraps Groq chat completions for grounded answer generation."""

    SYSTEM_PROMPT = """
        You are a document assistant.
        Answer only from provided context.
        If answer is not found say
        'I could not find that in the document.'
    """

    MULTI_QUERY_PROMPT = """
        Generate 3 alternate search queries for the user's question.

        Rules:
        - Keep meaning same
        - Use different wording
        - Return one query per line
        - No numbering

        Question:
        {query}
    """

    HYDE_PROMPT = """
        Write a short factual answer that might appear
        inside a document for the user's question.

        Rules:
        - 2 to 4 sentences
        - factual
        - concise
        - no explanation outside answer

        Question:
        {query}
    """

    def __init__(
        self,
        api_key: str = settings.GROQ_API_KEY,
        model: str = settings.GROQ_MODEL_NAME,
        temperature: float = settings.GROQ_TEMPERATURE,
        max_tokens: int = settings.GROQ_MAX_TOKENS,
        multi_query_temperature: float = settings.GROQ_MULTI_QUERY_TEMPERATURE,
        multi_query_max_tokens: int = settings.GROQ_MULTI_QUERY_MAX_TOKENS,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.multi_query_temperature = multi_query_temperature
        self.multi_query_max_tokens = multi_query_max_tokens

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

    def generate_queries(
        self,
        query: str
    ) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.MULTI_QUERY_PROMPT.format(query=query),
                }
            ],
            temperature=self.multi_query_temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content or ""

        queries = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        unique_queries = []

        for item in [query] + queries:
            if item not in unique_queries:
                unique_queries.append(item)

        return unique_queries[:4]

    def generate_hypothetical_answer(
        self,
        query: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.HYDE_PROMPT.format(
                        query=query
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=200,
        )

        content = (
            response.choices[0]
            .message
            .content
            or ""
        ).strip()

        logger.info(
            f"HyDE answer: {content[:120]}"
        )

        return content

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
