"""MCP server implementation for Ember."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        GetPromptResult,
        Prompt,
        ReadResourceResult,
        Resource,
        TextContent,
        Tool,
    )
except ImportError as exc:
    raise ImportError(
        "MCP is required for this integration. Install with: pip install mcp"
    ) from exc

from ember._internal.context import EmberContext, MetricsContext
from ember.api import models, operators
from ember.integrations.mcp.graph_advisor import (
    RunRequest,
    plan_session,
    run_session,
    schema_for_mcp_tool_request,
    schema_for_mcp_tool_response,
)
from ember.integrations.mcp.graph_advisor.errors import (
    GraphAdvisorError,
    RuntimeUnavailableError,
    SchemaInvalidError,
)
from ember.integrations.mcp.graph_advisor.schemas import ErrorPayload, Event
from ember.integrations.mcp.tools import ModelToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class EmberToolResult:
    """Result from Ember tool execution."""

    content: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


class EmberMCPServer:
    """MCP server exposing Ember capabilities.

    This server implements the Model Context Protocol to expose Ember's
    model orchestration capabilities as tools, resources, and prompts
    that can be accessed from any MCP-compatible client.

    Example:
        >>> from ember.integrations.mcp import EmberMCPServer
        >>>
        >>> # Create and run server
        >>> server = EmberMCPServer()
        >>> asyncio.run(server.run())
    """

    def __init__(self, name: str = "ember-mcp-server"):
        self.server = Server(name)
        self.context = EmberContext.current()
        self.metrics_context = MetricsContext()

        graph_config = self._load_graph_config()
        self._graph_enabled: bool = bool(graph_config.get("enabled", False))
        self._graph_allowed_clients: List[Dict[str, Any]] = list(
            graph_config.get("allowed_clients") or []
        )
        configured_models = graph_config.get("allowed_models")
        if configured_models:
            self._graph_allowed_models = list(configured_models)
        else:
            self._graph_allowed_models = models.list()
        self._graph_registry = self._build_graph_registry(self._graph_allowed_models)

        # Register capabilities
        self._register_tools()
        self._register_resources()
        self._register_prompts()

        logger.info(f"Initialized EmberMCPServer: {name}")

    def _register_tools(self):
        """Register Ember capabilities as MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Ember tools."""
            tools = [
                Tool(
                    name="ember_generate",
                    description="Generate text using any model in Ember's registry",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model identifier (e.g., claude-3-opus)",
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-1.0)",
                                "default": 0.7,
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                            },
                        },
                        "required": ["prompt", "model"],
                    },
                ),
                Tool(
                    name="ember_ensemble",
                    description="Run ensemble voting across multiple models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of model identifiers",
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["majority_vote", "weighted", "confidence"],
                                "description": "Voting strategy",
                                "default": "majority_vote",
                            },
                        },
                        "required": ["prompt", "models"],
                    },
                ),
                Tool(
                    name="ember_verify",
                    description="Verify and potentially improve model output",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Original prompt",
                            },
                            "output": {
                                "type": "string",
                                "description": "Output to verify",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model to use for verification",
                            },
                            "criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Verification criteria",
                            },
                        },
                        "required": ["prompt", "output", "model"],
                    },
                ),
                Tool(
                    name="ember_compare_models",
                    description="Compare outputs from multiple models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Models to compare",
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include performance metrics",
                                "default": True,
                            },
                        },
                        "required": ["prompt", "models"],
                    },
                ),
                Tool(
                    name="ember_stream",
                    description="Stream text generation from a model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The input prompt",
                            },
                            "model": {
                                "type": "string",
                                "description": "Model identifier",
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "Approximate size of each chunk",
                                "default": 10,
                            },
                        },
                        "required": ["prompt", "model"],
                    },
                ),
            ]

            if self._graph_enabled:
                tools.extend(
                    [
                        Tool(
                            name="ember_graph_advisor.plan",
                            description="Validate and normalize a compact NON graph without execution.",
                            inputSchema=schema_for_mcp_tool_request(),
                        ),
                        Tool(
                            name="ember_graph_advisor.run",
                            description="Execute a compact NON graph with judge synthesis.",
                            inputSchema=schema_for_mcp_tool_request(),
                            outputSchema=schema_for_mcp_tool_response(),
                        ),
                        Tool(
                            name="ember_graph_advisor.models",
                            description="List models currently available to the Graph Advisor.",
                            inputSchema={
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    ]
                )

            return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> CallToolResult:
            """Execute an Ember tool."""
            if arguments is None:
                arguments = {}

            try:
                if name == "ember_generate":
                    result = await self._tool_generate(**arguments)
                elif name == "ember_ensemble":
                    result = await self._tool_ensemble(**arguments)
                elif name == "ember_verify":
                    result = await self._tool_verify(**arguments)
                elif name == "ember_compare_models":
                    result = await self._tool_compare_models(**arguments)
                elif name == "ember_stream":
                    result = await self._tool_stream(**arguments)
                elif name == "ember_graph_advisor.plan":
                    result = await self._tool_graph_advisor_plan(arguments)
                elif name == "ember_graph_advisor.run":
                    result = await self._tool_graph_advisor_run(arguments)
                elif name == "ember_graph_advisor.models":
                    result = await self._tool_graph_advisor_models()
                else:
                    result = EmberToolResult(content=f"Unknown tool: {name}", is_error=True)

                return CallToolResult(
                    content=[TextContent(type="text", text=result.content)],
                    isError=result.is_error,
                )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )

    def _load_graph_config(self) -> Dict[str, Any]:
        """Return MCP Graph Advisor configuration from the active context."""

        config = self.context.get_config("mcp.graph_advisor", {})
        if isinstance(config, dict):
            return config
        return {}

    def _build_graph_registry(self, model_ids: List[str]) -> ModelToolRegistry:
        """Instantiate model bindings for Graph Advisor tools."""

        registry = ModelToolRegistry()
        registered: List[str] = []
        for model_id in model_ids:
            try:
                binding = models.instance(model_id)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize model '%s' for Graph Advisor: %s",
                    model_id,
                    exc,
                )
                continue
            registry.register(model_id, binding)
            registered.append(model_id)
        if registered:
            self._graph_allowed_models = registered
        return registry

    @staticmethod
    def _graph_error_payload(exc: GraphAdvisorError) -> ErrorPayload:
        """Convert a GraphAdvisorError into a serialisable payload."""

        return ErrorPayload(code=exc.code, message=str(exc), remediation=exc.remediation)

    async def _tool_graph_advisor_models(self) -> EmberToolResult:
        """Return the list of models available to the Graph Advisor."""

        if not self._graph_enabled:
            return EmberToolResult(content="Graph Advisor is disabled.", is_error=True)

        allowed = self._graph_allowed_models
        payload = {"allowed_models": allowed}
        return EmberToolResult(content=json.dumps(payload, indent=2), metadata=payload)

    async def _tool_graph_advisor_plan(self, payload: Dict[str, Any]) -> EmberToolResult:
        """Validate and normalize a compact NON graph without execution."""

        if not self._graph_enabled:
            return EmberToolResult(content="Graph Advisor is disabled.", is_error=True)

        try:
            request = RunRequest.model_validate(payload)
        except ValidationError as exc:
            error = SchemaInvalidError(str(exc))
            return EmberToolResult(
                content=json.dumps(self._graph_error_payload(error).model_dump(), indent=2),
                is_error=True,
            )

        try:
            _, response = plan_session(request)
        except GraphAdvisorError as exc:
            return EmberToolResult(
                content=json.dumps(self._graph_error_payload(exc).model_dump(), indent=2),
                is_error=True,
            )

        return EmberToolResult(
            content=json.dumps(response.model_dump(), indent=2),
            metadata={"status": "planned"},
        )

    async def _tool_graph_advisor_run(self, payload: Dict[str, Any]) -> EmberToolResult:
        """Execute a compact NON graph and return aggregated events."""

        if not self._graph_enabled:
            return EmberToolResult(content="Graph Advisor is disabled.", is_error=True)

        try:
            request = RunRequest.model_validate(payload)
        except ValidationError as exc:
            error = SchemaInvalidError(str(exc))
            return EmberToolResult(
                content=json.dumps(self._graph_error_payload(error).model_dump(), indent=2),
                is_error=True,
            )

        try:
            session, plan_preview = plan_session(request)
        except GraphAdvisorError as exc:
            return EmberToolResult(
                content=json.dumps(self._graph_error_payload(exc).model_dump(), indent=2),
                is_error=True,
            )

        events: List[Dict[str, Any]] = []

        async def emit(event: Event) -> None:
            events.append(event.model_dump())

        try:
            response = await run_session(session, request, self._graph_registry, emit)
        except GraphAdvisorError as exc:
            error_payload = self._graph_error_payload(exc)
            events.append(
                Event(
                    event="error",
                    sequence=len(events),
                    session_id=session.session_id,
                    data=error_payload.model_dump(),
                ).model_dump()
            )
            return EmberToolResult(
                content=json.dumps({"events": events, "error": error_payload.model_dump()}, indent=2),
                is_error=True,
            )
        except Exception as exc:
            runtime_error = RuntimeUnavailableError(str(exc))
            error_payload = self._graph_error_payload(runtime_error)
            events.append(
                Event(
                    event="error",
                    sequence=len(events),
                    session_id=session.session_id,
                    data=error_payload.model_dump(),
                ).model_dump()
            )
            return EmberToolResult(
                content=json.dumps({"events": events, "error": error_payload.model_dump()}, indent=2),
                is_error=True,
            )

        payload_out = {
            "plan": plan_preview.model_dump(),
            "events": events,
            "response": response.model_dump(),
        }
        return EmberToolResult(
            content=json.dumps(payload_out, indent=2),
            metadata={"session_id": session.session_id},
        )

    async def _tool_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> EmberToolResult:
        """Generate text using Ember."""
        try:
            ember_model = models.get_model(model)

            # Build parameters
            params = {"temperature": temperature}
            if max_tokens:
                params["max_tokens"] = max_tokens

            # Generate response
            with self.metrics_context.track(model=model) as span:
                response = await asyncio.to_thread(ember_model.generate, prompt, **params)
                span.record_response(response, model=model)

            # Get metrics
            metrics = self.metrics_context.get_last_metrics()

            return EmberToolResult(
                content=(response.content if hasattr(response, "content") else str(response)),
                metadata={
                    "model": model,
                    "usage": metrics.get("usage", {}),
                    "latency_ms": metrics.get("latency_ms", 0),
                },
            )

        except Exception as e:
            return EmberToolResult(content=f"Generation failed: {str(e)}", is_error=True)

    async def _tool_ensemble(
        self, prompt: str, models: List[str], strategy: str = "majority_vote"
    ) -> EmberToolResult:
        """Run ensemble across multiple models."""
        try:
            # Create operators for each model
            ops = [operators.Operator(model=m) for m in models]

            # Create ensemble
            ensemble = operators.EnsembleOperator(operators=ops, strategy=strategy)

            # Run ensemble
            result = await asyncio.to_thread(ensemble.run, prompt)

            # Format response
            response_data = {
                "consensus": result.content,
                "votes": result.metadata.get("votes", {}),
                "confidence": result.metadata.get("confidence", 0.0),
                "models_used": models,
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(content=f"Ensemble failed: {str(e)}", is_error=True)

    async def _tool_verify(
        self, prompt: str, output: str, model: str, criteria: Optional[List[str]] = None
    ) -> EmberToolResult:
        """Verify and improve output."""
        try:
            verifier = operators.VerifierOperator(model=model, criteria=criteria or [])

            result = await asyncio.to_thread(verifier.verify, prompt, output)

            response_data = {
                "is_valid": result.is_valid,
                "issues": result.issues,
                "improved_output": (result.improved_output if not result.is_valid else output),
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(content=f"Verification failed: {str(e)}", is_error=True)

    async def _tool_compare_models(
        self, prompt: str, models: List[str], include_metrics: bool = True
    ) -> EmberToolResult:
        """Compare outputs from multiple models."""
        try:
            results = {}

            for model_name in models:
                ember_model = models.get_model(model_name)

                with self.metrics_context.track(model=model_name) as span:
                    response = await asyncio.to_thread(ember_model.generate, prompt)
                    span.record_response(response, model=model_name)

                metrics = self.metrics_context.get_last_metrics()

                results[model_name] = {
                    "output": (response.content if hasattr(response, "content") else str(response)),
                    "metrics": metrics if include_metrics else {},
                }

            return EmberToolResult(
                content=json.dumps(results, indent=2), metadata={"comparison": results}
            )

        except Exception as e:
            return EmberToolResult(content=f"Comparison failed: {str(e)}", is_error=True)

    async def _tool_stream(self, prompt: str, model: str, chunk_size: int = 10) -> EmberToolResult:
        """Stream text generation."""
        try:
            ember_model = models.get_model(model)
            chunks = []

            # Collect chunks (in real implementation, this would stream)
            async for chunk in ember_model.astream(prompt):
                chunks.append(chunk)

            # Format as streaming response
            response_data = {
                "model": model,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "complete_text": "".join(chunks),
            }

            return EmberToolResult(
                content=json.dumps(response_data, indent=2), metadata=response_data
            )

        except Exception as e:
            return EmberToolResult(content=f"Streaming failed: {str(e)}", is_error=True)

    def _register_resources(self):
        """Register Ember data as MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available Ember resources."""
            return [
                Resource(
                    uri="ember://models/registry",
                    name="Model Registry",
                    description="Complete list of available models with capabilities and pricing",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://models/costs",
                    name="Model Costs",
                    description="Current pricing information for all models",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://metrics/usage",
                    name="Usage Metrics",
                    description="Current session usage statistics",
                    mimeType="application/json",
                ),
                Resource(
                    uri="ember://operators/types",
                    name="Operator Types",
                    description="Available operator types and their descriptions",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> ReadResourceResult:
            """Read an Ember resource."""
            try:
                if uri == "ember://models/registry":
                    content = await self._get_model_registry()
                elif uri == "ember://models/costs":
                    content = await self._get_model_costs()
                elif uri == "ember://metrics/usage":
                    content = await self._get_usage_metrics()
                elif uri == "ember://operators/types":
                    content = await self._get_operator_types()
                else:
                    content = f"Unknown resource: {uri}"

                return ReadResourceResult(
                    contents=[TextContent(type="text", text=content, mimeType="application/json")]
                )

            except Exception as e:
                logger.error(f"Resource read failed: {e}")
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": str(e)}),
                            mimeType="application/json",
                        )
                    ]
                )

    async def _get_model_registry(self) -> str:
        """Get model registry information."""
        registry = models.get_registry()

        formatted_registry = {}
        for model_id, info in registry.items():
            formatted_registry[model_id] = {
                "provider": info.provider,
                "capabilities": info.capabilities,
                "context_length": info.context_length,
                "supports_streaming": info.supports_streaming,
                "supports_tools": info.supports_tools,
            }

        return json.dumps(formatted_registry, indent=2)

    async def _get_model_costs(self) -> str:
        """Get model cost information."""
        registry = models.get_registry()

        costs = {}
        for model_id, info in registry.items():
            costs[model_id] = {
                "input_cost_per_1m": info.input_cost,
                "output_cost_per_1m": info.output_cost,
                "currency": "USD",
            }

        return json.dumps(costs, indent=2)

    async def _get_usage_metrics(self) -> str:
        """Get current usage metrics."""
        # This would aggregate real metrics in production
        metrics = {
            "session_start": datetime.now().isoformat(),
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "models_used": [],
            "average_latency_ms": 0,
        }

        return json.dumps(metrics, indent=2)

    async def _get_operator_types(self) -> str:
        """Get available operator types."""
        operator_info = {
            "Operator": {
                "description": "Base operator for single model calls",
                "use_case": "Simple text generation tasks",
            },
            "EnsembleOperator": {
                "description": "Combines multiple models with voting strategies",
                "use_case": "High-stakes decisions requiring consensus",
            },
            "VerifierOperator": {
                "description": "Verifies and improves model outputs",
                "use_case": "Quality assurance and output validation",
            },
            "JudgeSynthesisOperator": {
                "description": "Judges and synthesizes multiple outputs",
                "use_case": "Complex reasoning and synthesis tasks",
            },
        }

        return json.dumps(operator_info, indent=2)

    def _register_prompts(self):
        """Register Ember prompt templates."""

        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="code_review",
                    description="Comprehensive code review with multiple perspectives",
                    arguments=[
                        {
                            "name": "code",
                            "description": "Code to review",
                            "required": True,
                        },
                        {
                            "name": "language",
                            "description": "Programming language",
                            "required": False,
                        },
                    ],
                ),
                Prompt(
                    name="chain_of_thought",
                    description="Step-by-step reasoning for complex problems",
                    arguments=[
                        {
                            "name": "problem",
                            "description": "Problem to solve",
                            "required": True,
                        }
                    ],
                ),
                Prompt(
                    name="compare_approaches",
                    description="Compare multiple approaches to a problem",
                    arguments=[
                        {
                            "name": "problem",
                            "description": "Problem description",
                            "required": True,
                        },
                        {
                            "name": "approaches",
                            "description": "List of approaches",
                            "required": True,
                        },
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: Optional[Dict[str, str]] = None
        ) -> GetPromptResult:
            """Get a specific prompt template."""
            if arguments is None:
                arguments = {}

            try:
                if name == "code_review":
                    messages = self._prompt_code_review(**arguments)
                elif name == "chain_of_thought":
                    messages = self._prompt_chain_of_thought(**arguments)
                elif name == "compare_approaches":
                    messages = self._prompt_compare_approaches(**arguments)
                else:
                    messages = [{"role": "user", "content": f"Unknown prompt: {name}"}]

                return GetPromptResult(description=f"Prompt template: {name}", messages=messages)

            except Exception as e:
                logger.error(f"Prompt generation failed: {e}")
                return GetPromptResult(
                    description="Error generating prompt",
                    messages=[{"role": "user", "content": f"Error: {str(e)}"}],
                )

    def _prompt_code_review(self, code: str, language: str = "python") -> List[Dict[str, str]]:
        """Generate code review prompt."""
        return [
            {
                "role": "system",
                "content": f"You are an expert {language} code reviewer. Review code for:\n"
                "1. Correctness and logic errors\n"
                "2. Performance and efficiency\n"
                "3. Security vulnerabilities\n"
                "4. Code style and best practices\n"
                "5. Maintainability and documentation",
            },
            {
                "role": "user",
                "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```",
            },
        ]

    def _prompt_chain_of_thought(self, problem: str) -> List[Dict[str, str]]:
        """Generate chain of thought prompt."""
        return [
            {
                "role": "system",
                "content": "You are a logical reasoning expert. Break down problems step by step, "
                "showing your work at each stage. Be thorough and explicit in your reasoning.",
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\n"
                "Please solve this step by step:\n"
                "1. First, identify what we know\n"
                "2. Then, determine what we need to find\n"
                "3. Next, outline the approach\n"
                "4. Finally, work through the solution\n"
                "5. Verify the answer",
            },
        ]

    def _prompt_compare_approaches(self, problem: str, approaches: str) -> List[Dict[str, str]]:
        """Generate comparison prompt."""
        return [
            {
                "role": "system",
                "content": (
                    "You are an analytical expert who excels at comparing different approaches. "
                    "Provide balanced, objective analysis considering multiple factors."
                ),
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\n"
                f"Approaches to compare:\n{approaches}\n\n"
                "Please analyze each approach considering:\n"
                "- Effectiveness\n"
                "- Efficiency\n"
                "- Cost\n"
                "- Complexity\n"
                "- Risks\n"
                "- Long-term implications\n\n"
                "Provide a recommendation with rationale.",
            },
        ]

    async def run(self, transport: str = "stdio"):
        """Run the MCP server.

        Args:
            transport: Transport method ("stdio" or "http")
        """
        logger.info(f"Starting EmberMCPServer with {transport} transport")

        if transport == "stdio":
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(server_name="ember-mcp", server_version="1.0.0"),
                )
        elif transport == "http":
            # HTTP transport would be implemented here
            raise NotImplementedError("HTTP transport not yet implemented")
        else:
            raise ValueError(f"Unknown transport: {transport}")
