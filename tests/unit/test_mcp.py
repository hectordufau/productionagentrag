from __future__ import annotations

import asyncio

import pytest

from agentic_rag.agentic import AgenticRAGGraph, analyze_query
from agentic_rag.mcp import MCPToolClient, MCPToolError, create_mcp_server
from agentic_rag.retrieval import DeterministicHashEmbedding, VectorStore
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore


def setup_store() -> tuple[InMemoryStore, Document]:
    store = InMemoryStore()
    document = Document.from_content("demo", "guide.txt", "text/plain", "Qdrant stores vectors")
    store.add_document(document)
    return store, document


def test_server_client_returns_metadata_without_content():
    store, document = setup_store()
    result = MCPToolClient(create_mcp_server(store)).call("get_document_metadata", {"document_id": document.document_id})
    assert result["document"]["filename"] == "guide.txt"
    assert "content" not in result["document"]


def test_allowlist_invalid_arguments_and_limit_fail_closed():
    store, document = setup_store()
    client = MCPToolClient(create_mcp_server(store), max_calls=1)
    with pytest.raises(MCPToolError, match="allowlisted"):
        client.call("delete_document", {"document_id": document.document_id})
    with pytest.raises(MCPToolError, match="invalid"):
        client.call("get_document_metadata", {"document_id": 3})
    client.call("get_document_metadata", {"document_id": document.document_id})
    with pytest.raises(MCPToolError, match="limit"):
        client.call("get_document_metadata", {"document_id": document.document_id})


def test_timeout_and_tool_failure_are_explicit():
    store, _document = setup_store()
    server = create_mcp_server(store)
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
    server.add_tool(slow, name="slow_tool")
    client = MCPToolClient(server, timeout_s=0.001)
    with pytest.raises(MCPToolError, match="allowlisted"):
        client.call("slow_tool", {})
    with pytest.raises(MCPToolError, match="failed"):
        MCPToolClient(create_mcp_server(store)).call("get_document_metadata", {"document_id": "missing"})


def test_routing_modes_and_tool_provenance():
    store, document = setup_store()
    assert analyze_query("where are vectors stored")["route_mode"] == "retrieval_only"
    assert analyze_query(f"what is metadata for document {document.document_id}")["route_mode"] == "tool_only"
    assert analyze_query(f"what source cites document {document.document_id}")["route_mode"] == "retrieval_plus_tool"
    graph = AgenticRAGGraph(store, VectorStore(DeterministicHashEmbedding()), None, mcp_client=MCPToolClient(create_mcp_server(store)))
    state = graph.run(f"what is the metadata for document {document.document_id}")
    assert state["status"] == "OK"
    assert state["citations"][0]["kind"] == "tool"
    assert state["metrics"]["tool_calls"] == 1
    assert state["decision_trace"][-1]["status"] == "success"
    assert "system prompt" not in state["answer"].lower()
