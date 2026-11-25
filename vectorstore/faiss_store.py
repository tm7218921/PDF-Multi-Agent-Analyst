
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import faiss


class FAISSVectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product / cosine similarity when
    embeddings are L2-normalised) and keeps a parallel list of metadata.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []  # parallel to FAISS vectors

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their pre-computed embeddings.

        chunks      – list of dicts with keys: text, doc_name, page_number, chunk_index
        embeddings  – float32 array of shape (len(chunks), embedding_dim)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )
        if len(chunks) == 0:
            return

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for the top_k most similar chunks.

        query_embedding – float32 array of shape (1, embedding_dim) or (embedding_dim,)
        Returns list of chunk dicts enriched with a 'score' key.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Vector store is empty. Please upload and process PDFs first."
            )

        query = np.atleast_2d(query_embedding).astype(np.float32)
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, actual_k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def reset(self) -> None:
        """Clear the index and all metadata (e.g. on new upload)."""
        self.index = None
        self.metadata = []
