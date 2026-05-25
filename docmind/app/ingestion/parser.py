import logging
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class Parser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        suffix = Path(self.file_path).suffix.lower()

        if suffix == '.pdf':
            loader = PDFLoader(self.file_path)
            return loader.load()
        elif suffix == '.txt':
            loader = TXTLoader(self.file_path)
            return loader.load()
        elif suffix == '.md':
            loader = MDLoader(self.file_path)
            return loader.load()
        elif suffix == '.docx':
            loader = DOCXLoader(self.file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")
    

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
            logger.warning("pdfplumber returned empty text — falling back to pypdf")
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
    

class TXTLoader:
    def __init__(self, txt_path):
        self.txt_path = Path(txt_path)
        if not self.txt_path.exists():
            raise FileNotFoundError(f"TXT not found: {txt_path}")
        logger.info(f"TXTLoader initialized → {self.txt_path.name}")

    def load(self) -> List[Tuple[int, str]]:
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [(1, text)]
    

class MDLoader:
    def __init__(self, md_path):
        self.md_path = Path(md_path)
        if not self.md_path.exists():
            raise FileNotFoundError(f"MD not found: {md_path}")
        logger.info(f"MDLoader initialized → {self.md_path.name}")

    def load(self) -> List[Tuple[int, str]]:
        with open(self.md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [(1, text)]


class DOCXLoader:
    def __init__(self, docx_path):
        self.docx_path = Path(docx_path)
        if not self.docx_path.exists():
            raise FileNotFoundError(f"DOCX not found: {docx_path}")
        logger.info(f"DOCXLoader initialized → {self.docx_path.name}")

    def load(self) -> List[Tuple[int, str]]:        
        document = Document(self.docx_path)
        text = "\n".join(
            para.text
            for para in document.paragraphs
            if para.text.strip()
        )
        return [(1, text)]
