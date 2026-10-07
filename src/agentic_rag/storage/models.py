from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    filename: str
    mime_type: str
    checksum: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_content(
        cls,
        source: str,
        filename: str,
        mime_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Document":
        checksum = sha256(content.encode("utf-8")).hexdigest()
        return cls(
            checksum, source, filename, mime_type, checksum, content, metadata=metadata or {}
        )


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    method: str
