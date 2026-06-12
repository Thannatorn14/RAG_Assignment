import re
import pdfplumber
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """Returns list of {page, text} dicts."""
    pages = []
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="replace")
    # Treat entire file as page 1
    return [{"page": 1, "text": text}]


def chunk_pages(pages: list[dict], chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split page texts into overlapping word-level chunks."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"page": page["page"], "text": chunk_text})
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks
