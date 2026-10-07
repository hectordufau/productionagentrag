# MCP tool integration

This project uses the official Python MCP SDK `mcp==2.3.0` (SDK 2.x `MCPServer` API). The standalone server is `agentic_rag.mcp.server`; its only exposed tool is the allowlisted `get_document_metadata(document_id)`, backed by the existing `InMemoryStore` document model. It never returns document content.

`MCPToolClient` is a fail-closed policy boundary: it validates the exact argument shape, permits only the allowlist, limits a request to two calls, applies a timeout, and turns transport/tool/output failures into explicit `MCPToolError`s. Tool responses are untrusted data. The graph never interprets metadata as instructions, and metadata is not added to the generation prompt.

Agentic routing remains one LangGraph architecture. Deterministic analysis selects:

- `retrieval_only`: normal baseline retrieval and generation.
- `tool_only`: metadata question with a document identifier; returns a tool-provenance answer without retrieval.
- `retrieval_plus_tool`: source/citation question that has both retrievable evidence and metadata; preserves chunk citations and appends identifiable tool provenance.

Example setup:

```python
from agentic_rag.mcp import MCPToolClient, create_mcp_server
client = MCPToolClient(create_mcp_server(store), timeout_s=2.0)
state = graph.run("what is the metadata for document abc123")
```

Run the focused demonstration and tests with `pytest -q tests/unit/test_mcp.py tests/unit/test_agentic.py`; the full gates remain `pytest -q` and `ruff check src tests scripts`.
