from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore


def test_duplicate_document_is_not_added_twice() -> None:
    store = InMemoryStore()
    doc = Document.from_content("sample", "a.txt", "text/plain", "qdrant vectors")
    assert store.add_document(doc) is True
    assert store.add_document(doc) is False


def test_baseline_returns_citations_for_grounded_query() -> None:
    store = InMemoryStore()
    doc = Document.from_content("sample", "a.txt", "text/plain", "qdrant stores vectors")
    store.add_document(doc)
    store.add_chunks(chunk_document(doc))
    result = RetrievalService(store).answer("where qdrant vectors")
    assert result.grounded is True
    assert result.citations[0].document_id == doc.document_id


def test_baseline_fails_closed_without_context() -> None:
    result = RetrievalService(InMemoryStore()).answer("unknown")
    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.grounded is False
