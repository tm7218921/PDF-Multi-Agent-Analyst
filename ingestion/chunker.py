"""
Text chunking with configurable chunk_size and chunk_overlap.
Operates on the page-level output from pdf_loader.
"""
from __future__ import annotations

from typing import List, Dict

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits a plain-text string into overlapping chunks by character count.
    Attempts to break on whitespace boundaries to keep words intact.
    """
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        start += step

    return chunks


def chunk_text(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Takes a list of page dicts from pdf_loader and returns a flat list of
    chunk dicts:
        {
            "text":        str,
            "doc_name":    str,
            "page_number": int,
            "chunk_index": int,
        }
    """
    all_chunks: List[Dict] = []

    for page in pages:
        text = page["text"]
        doc_name = page["doc_name"]
        page_number = page["page_number"]

        sub_chunks = _split_text(text, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(sub_chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_index": idx,
                }
            )

    return all_chunks
