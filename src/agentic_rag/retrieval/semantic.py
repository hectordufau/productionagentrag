"""Local semantic embeddings backed by sentence-transformers.

The dependency and model are intentionally explicit: importing this module does not
silently download or fall back to the deterministic control provider.
"""
from __future__ import annotations

from typing import Any

from .embedding import EmbeddingProvider


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence-transformers provider using a small Apache-2.0 model."""

    name = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "main"
    license = "Apache-2.0"
    dimension = 384
    normalized = True
    distance = "Cosine"

    def __init__(self, model_name: str = name, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "semantic embeddings require the optional sentence-transformers dependency"
            ) from exc
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, **kwargs)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        values = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return values.astype("float32").tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        values = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return values.astype("float32").tolist()
