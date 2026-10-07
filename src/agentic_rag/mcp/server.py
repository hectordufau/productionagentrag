"""A small official MCP SDK server exposing only document metadata."""
from __future__ import annotations

from typing import Any

from agentic_rag.storage.store import InMemoryStore
from mcp.server.mcpserver import MCPServer

DOCUMENT_METADATA_TOOL = "get_document_metadata"


class DocumentMetadataServer:
    """Own the MCP server and its explicitly allowlisted metadata tool."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.server = MCPServer(name="production-agentic-rag-metadata", version="0.1.0")
        self.server.add_tool(
            self.get_document_metadata,
            name=DOCUMENT_METADATA_TOOL,
            description="Return non-content metadata for one document.",
            structured_output=True,
        )

    def get_document_metadata(self, document_id: str) -> dict[str, Any]:
        if not isinstance(document_id, str) or not document_id or len(document_id) > 256:
            raise ValueError("document_id must be a non-empty string of at most 256 characters")
        document = self.store.documents.get(document_id)
        if document is None:
            raise KeyError(f"document not found: {document_id}")
        # Deliberately exclude content: tool output is data, never instructions.
        return {
            "document_id": document.document_id,
            "source": document.source,
            "filename": document.filename,
            "mime_type": document.mime_type,
            "checksum": document.checksum,
            "created_at": document.created_at.isoformat(),
            "metadata": dict(document.metadata),
        }


def create_mcp_server(store: InMemoryStore) -> MCPServer:
    """Create the standalone official-SDK MCP server."""
    return DocumentMetadataServer(store).server
