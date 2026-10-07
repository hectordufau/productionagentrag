from agentic_rag.chunking.core import ChunkingConfig, chunk_document
from agentic_rag.storage.models import Document


def test_document_checksum_is_deterministic() -> None:
    a = Document.from_content("x", "a.txt", "text/plain", "hello")
    b = Document.from_content("x", "a.txt", "text/plain", "hello")
    assert a.document_id == b.document_id == a.checksum


def test_chunking_is_bounded_and_overlapping() -> None:
    doc = Document.from_content("x", "a.txt", "text/plain", " ".join(f"w{i}" for i in range(10)))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=4, overlap=1))
    assert [c.content for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
