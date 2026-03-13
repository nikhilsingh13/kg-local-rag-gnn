"""
Converts raw PDF files into clean text and overlapping chunks.

Pipeline:
    PDF file -> raw text -> clean text -> List[Chunk]

Each Chunk carries:
    - text        : the chunk content
    - paper_id    : arXiv ID derived from filename
    - chunk_index : position within the paper
    - page_range  : (start_page, end_page) the chunk spans
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
import fitz


@dataclass
class Chunk:
    text: str
    paper_id: str
    chunk_index: int
    page_range: tuple[int, int]
    metadata: dict = field(default_factory=dict)


class PDFExtractor:
    """Extract and chunk text from a PDF file."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """
        Args:
            chunk_size:    Approximate token count per chunk (words used as proxy).
            chunk_overlap: Number of words shared between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from a PDF, preserving paragraph structure.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Full text of the document.
        """
        text_pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_pages.append(page.get_text("text"))
            
        return "\n".join(text_pages)

    def clean_text(self, raw_text: str) -> str:
        """Remove noise: hyphenation, extra whitespace.

        Args:
            raw_text: Text as extracted from PDF.

        Returns:
            Cleaned text suitable for chunking.
        """
        # Fix hyphenated words split across lines (e.g., "know-\nledge" -> "knowledge")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
        
        # Reduce multiple sequential newlines into a standard double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Strip leading and trailing whitespace from every line
        text = "\n".join(line.strip() for line in text.split("\n"))
        
        return text

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        """Split text into overlapping word-level chunks.

        Args:
            text:     Cleaned document text.
            paper_id: Identifier (e.g. arXiv ID like "1609.02907").

        Returns:
            Ordered list of Chunk objects.
        """
        words = text.split()
        chunks = []
        
        step = self.chunk_size - self.chunk_overlap
        step = max(1, step)  # Ensure step is strictly positive

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                paper_id=paper_id,
                chunk_index=len(chunks),
                # Mapping exact page ranges per word chunk is complex; 
                # leaving as (0, 0) placeholder for the MVP pipeline.
                page_range=(0, 0)
            ))
            
            # Break early if the final chunk captures the very end of the document
            if i + self.chunk_size >= len(words):
                break
                
        return chunks

    def process(self, pdf_path: Path) -> list[Chunk]:
        """Full pipeline: PDF -> cleaned chunks.

        Args:
            pdf_path: Path to the PDF.

        Returns:
            List of Chunk objects ready for embedding + entity extraction.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
            
        paper_id = pdf_path.stem  # e.g. "1609.02907v4"
        
        raw_text = self.extract_text(pdf_path)
        cleaned_text = self.clean_text(raw_text)
        chunks = self.chunk(cleaned_text, paper_id)
        
        return chunks