"""Model Context Protocol (MCP) integration for Ember.

This module provides MCP server implementations that expose Ember's capabilities
as tools, resources, and prompts accessible from any MCP-compatible client.
"""

try:  # pragma: no cover - optional dependency
    from .server import EmberMCPServer
except ImportError:  # pragma: no cover - MCP extras not installed
    EmberMCPServer = None

__all__ = ["EmberMCPServer"]
