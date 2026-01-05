"""Ember integrations with popular AI frameworks.

This module provides integrations with:
- DSPy: Declarative self-improving language model programs
- OpenAI Swarm: Lightweight multi-agent orchestration
- Anthropic MCP: Model Context Protocol for universal AI connectivity
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ember.integrations.dspy import EmberLM
    from ember.integrations.mcp import EmberMCPServer
    from ember.integrations.swarm import EmberSwarmClient

__all__ = ["EmberLM", "EmberSwarmClient", "EmberMCPServer"]
