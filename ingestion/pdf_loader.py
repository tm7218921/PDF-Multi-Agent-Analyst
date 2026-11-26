"""
PDF text extraction using PyMuPDF (fitz).
Returns a list of dicts: {text, page_number, doc_name}
"""
from __future__ import annotations

import fitz
from typing import List, Dict


def load_pdf(file_bytes: bytes, doc_name: str) -> List[Dict]:
    """
    Extract text from a PDF given its raw bytes and an identifying name.

    Returns a list of page dicts:
        [{"text": str, "page_number": int, "doc_name": str}, ...]

    Raises ValueError for empty or un-extractable PDFs.
    """
    if not file_bytes:
        raise ValueError(f"PDF '{doc_name}' is empty (no bytes received).")

    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{doc_name}': {exc}") from exc

    if pdf_doc.page_count == 0:
        pdf_doc.close()
        raise ValueError(f"PDF '{doc_name}' has no pages.")

    pages: List[Dict] = []
    for page_index in range(pdf_doc.page_count):
        page = pdf_doc[page_index]
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append(
                {
                    "text": text,
                    "page_number": page_index + 1,
                    "doc_name": doc_name,
                }
            )

    pdf_doc.close()

    if not pages:
        raise ValueError(
            f"PDF '{doc_name}' contains no extractable text. "
            "It may be a scanned image-only PDF."
        )

    return pages
