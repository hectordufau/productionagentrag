"""MCP server and client integration for document metadata."""

from .client import MCPToolClient, MCPToolError
from .server import DOCUMENT_METADATA_TOOL, DocumentMetadataServer, create_mcp_server

__all__ = ["DOCUMENT_METADATA_TOOL", "DocumentMetadataServer", "MCPToolClient", "MCPToolError", "create_mcp_server"]
