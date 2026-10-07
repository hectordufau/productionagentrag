"""Fail-closed client policy around the official MCP server."""
from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from .server import DOCUMENT_METADATA_TOOL


class MCPToolError(RuntimeError):
    """Explicit, safe MCP failure."""


class MCPToolClient:
    """A bounded client facade; unknown tools and malformed results fail closed."""

    ALLOWLIST = frozenset({DOCUMENT_METADATA_TOOL})

    def __init__(self, server: MCPServer, *, timeout_s: float = 2.0, max_calls: int = 2) -> None:
        if timeout_s <= 0 or max_calls < 1:
            raise ValueError("timeout_s must be positive and max_calls must be positive")
        self.server = server
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self.calls = 0

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        return await asyncio.wait_for(self.server.call_tool(name, arguments), self.timeout_s)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.ALLOWLIST:
            raise MCPToolError("tool is not allowlisted")
        if self.calls >= self.max_calls:
            raise MCPToolError("MCP tool call limit exceeded")
        if not isinstance(arguments, dict) or set(arguments) != {"document_id"}:
            raise MCPToolError("invalid tool arguments")
        document_id = arguments.get("document_id")
        if not isinstance(document_id, str) or not document_id or len(document_id) > 256:
            raise MCPToolError("invalid document_id")
        self.calls += 1
        try:
            result = asyncio.run(self._call(name, arguments))
        except Exception as exc:
            raise MCPToolError(f"MCP tool failed: {type(exc).__name__}") from exc
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict) and structured:
            value: Any = structured
        else:
            content = getattr(result, "content", None)
            value = content if isinstance(content, dict) else None
        if not isinstance(value, dict) or value.get("document_id") != document_id:
            raise MCPToolError("invalid or untrusted MCP output")
        return {"tool": name, "document": dict(value)}
