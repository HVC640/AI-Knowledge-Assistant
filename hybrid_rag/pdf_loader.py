import logging
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads and extracts text from a PDF file.

    Uses pdfplumber first and falls back to pypdf if needed.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info(f"PDFLoader initialized → {self.pdf_path.name}")

    def load(self) -> List[Tuple[int, str]]:
        pages = self._load_with_pdfplumber()
        if not any(text.strip() for _, text in pages):
            logger.warning(
                "pdfplumber returned empty text — falling back to pypdf")
            pages = self._load_with_pypdf()

        total_chars = sum(len(text) for _, text in pages)
        logger.info(f"Loaded {len(pages)} pages | {total_chars:,} characters")
        return pages

    def _load_with_pdfplumber(self) -> List[Tuple[int, str]]:
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((page_number, text))
        return pages

    def _load_with_pypdf(self) -> List[Tuple[int, str]]:
        pages = []
        reader = PdfReader(str(self.pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((page_number, text))
        return pages
