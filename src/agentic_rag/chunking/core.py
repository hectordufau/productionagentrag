import re
from dataclasses import dataclass

from agentic_rag.storage.models import Chunk, Document


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 800
    overlap: int = 120


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    if config.chunk_size <= 0 or config.overlap < 0 or config.overlap >= config.chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    words = re.findall(r"\S+", document.content)
    chunks: list[Chunk] = []
    step = config.chunk_size - config.overlap
    for position, start in enumerate(range(0, len(words), step)):
        content_words = words[start : start + config.chunk_size]
        if not content_words:
            break
        content = " ".join(content_words)
        chunks.append(
            Chunk(
                f"{document.document_id}:{position}",
                document.document_id,
                content,
                position,
                {"strategy": config.strategy, "source": document.source, "filename": document.filename, **document.metadata},
            )
        )
        if start + config.chunk_size >= len(words):
            break
    return chunks
