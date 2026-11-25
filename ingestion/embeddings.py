"""
Embedding generation using sentence-transformers.
Wraps the model in a singleton so it's loaded once per session.
"""
from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384-dim


class EmbeddingModel:
    """Singleton wrapper around a SentenceTransformer model."""

    _instance: "EmbeddingModel | None" = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(_MODEL_NAME)
        return cls._instance

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of strings.
        Returns a float32 numpy array of shape (len(texts), dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string. Returns shape (1, dim)."""
        return self.embed([query])
