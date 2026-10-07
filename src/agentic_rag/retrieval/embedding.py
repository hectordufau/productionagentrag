"""Deterministic, dependency-free embedding providers for local evaluation."""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


class EmbeddingProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class DeterministicHashEmbedding(EmbeddingProvider):
    """Stable signed feature hashing; no model download or network required."""

    name = "deterministic-hash"

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        vector = [0.0] * self.dimension
        for token in tokens:
            for feature in (token, f"{token[0]}:{token[-1]}"):
                digest = hashlib.sha256(feature.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
