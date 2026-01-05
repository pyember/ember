"""Basic MCP server example for Ember."""

import asyncio
import logging

from ember.integrations.mcp import EmberMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    """Run a basic Ember MCP server."""
    print("Starting Ember MCP Server...")
    print("This server exposes Ember's capabilities via the Model Context Protocol.")
    print("\nAvailable tools:")
    print("- ember_generate: Generate text with any Ember model")
    print("- ember_ensemble: Run ensemble voting across models")
    print("- ember_verify: Verify and improve outputs")
    print("- ember_compare_models: Compare outputs from multiple models")
    print("- ember_stream: Stream text generation")
    print("\nAvailable resources:")
    print("- ember://models/registry: List all available models")
    print("- ember://models/costs: Get pricing information")
    print("- ember://metrics/usage: View usage statistics")
    print("\nPress Ctrl+C to stop the server.\n")

    # Create and run server
    server = EmberMCPServer(name="ember-basic")

    try:
        await server.run(transport="stdio")
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
