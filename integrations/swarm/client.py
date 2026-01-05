"""OpenAI-compatible client for Swarm using Ember backend."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union

from ember._internal.context import MetricsContext
from ember.api import models

logger = logging.getLogger(__name__)


@dataclass
class ChatCompletionMessage:
    """OpenAI-compatible message format."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class Choice:
    """OpenAI-compatible choice format."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str


@dataclass
class Usage:
    """OpenAI-compatible usage format."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """OpenAI-compatible chat completion format."""

    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None


@dataclass
class StreamChoice:
    """OpenAI-compatible streaming choice format."""

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


@dataclass
class StreamChatCompletion:
    """OpenAI-compatible streaming completion format."""

    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = ""
    choices: List[StreamChoice] = field(default_factory=list)


class EmberSwarmClient:
    """OpenAI-compatible client for Swarm using Ember backend.

    This client provides a drop-in replacement for OpenAI's client in Swarm,
    enabling use of any Ember-supported model for agent orchestration.

    Args:
        default_model: Default model to use if not specified in calls

    Example:
        >>> from swarm import Swarm
        >>> from ember.integrations.swarm import EmberSwarmClient
        >>>
        >>> # Create Ember-backed client
        >>> client = EmberSwarmClient(default_model="claude-3-opus-20240229")
        >>> swarm = Swarm(client=client)
        >>>
        >>> # Use with agents
        >>> agent = Agent(
        ...     name="Assistant",
        ...     model="gpt-4",  # Can use any Ember model
        ...     instructions="You are helpful."
        ... )
    """

    def __init__(self, default_model: str = "gpt-4"):
        self.default_model = default_model
        self.metrics_context = MetricsContext()

        # Create nested structure to match OpenAI client
        self.chat = self.Chat(self)

        # Model registry cache
        self._model_cache = {}

        logger.info(f"Initialized EmberSwarmClient with default model: {default_model}")

    class Chat:
        """Chat namespace matching OpenAI client structure."""

        def __init__(self, client: "EmberSwarmClient"):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            """Completions namespace for chat operations."""

            def __init__(self, client: "EmberSwarmClient"):
                self.client = client

            def create(
                self,
                messages: List[Dict[str, Any]],
                model: Optional[str] = None,
                frequency_penalty: Optional[float] = None,
                logit_bias: Optional[Dict[str, float]] = None,
                logprobs: Optional[bool] = None,
                top_logprobs: Optional[int] = None,
                max_tokens: Optional[int] = None,
                n: Optional[int] = None,
                presence_penalty: Optional[float] = None,
                response_format: Optional[Dict[str, Any]] = None,
                seed: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                stream: Optional[bool] = False,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                user: Optional[str] = None,
                function_call: Optional[Union[str, Dict[str, str]]] = None,
                functions: Optional[List[Dict[str, Any]]] = None,
                **kwargs,
            ) -> Union[ChatCompletion, Iterator[StreamChatCompletion]]:
                """Create a chat completion using Ember.

                Args:
                    messages: List of message dictionaries
                    model: Model to use (defaults to client's default)
                    stream: Whether to stream the response
                    **kwargs: Additional parameters supported by Ember

                Returns:
                    ChatCompletion or iterator of StreamChatCompletion chunks
                """
                # Use provided model or default
                model_name = model or self.client.default_model

                # Get or create Ember model
                ember_model = self.client._get_ember_model(model_name)

                # Convert functions to tools format if needed
                if functions and not tools:
                    tools = [{"type": "function", "function": func} for func in functions]
                    if function_call:
                        tool_choice = function_call

                # Prepare Ember call parameters
                ember_params = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stop": stop,
                }

                # Add tool-related parameters if present
                if tools:
                    ember_params["tools"] = tools
                if tool_choice:
                    ember_params["tool_choice"] = tool_choice

                # Filter out None values
                ember_params = {k: v for k, v in ember_params.items() if v is not None}

                # Handle streaming
                if stream:
                    return self._stream_response(ember_model, ember_params, model_name)

                # Non-streaming response
                with self.client.metrics_context.track(model=model_name) as span:
                    try:
                        response = ember_model.complete(**ember_params)
                        span.record_response(response, model=model_name)
                        return self._convert_response(response, model_name)
                    except Exception as err:  # pragma: no cover - defensive logging
                        logger.error("Ember model call failed: %s", err)
                        raise RuntimeError(f"Model call failed: {err}") from err

            def _convert_response(self, ember_response: Any, model: str) -> ChatCompletion:
                """Convert Ember response to OpenAI format."""
                # Generate unique ID
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

                # Extract content
                content = ""
                if hasattr(ember_response, "content"):
                    content = ember_response.content
                elif isinstance(ember_response, dict) and "content" in ember_response:
                    content = ember_response["content"]
                elif isinstance(ember_response, str):
                    content = ember_response

                # Build message
                message = ChatCompletionMessage(role="assistant", content=content)

                # Handle tool calls if present
                if hasattr(ember_response, "tool_calls") and ember_response.tool_calls:
                    message.tool_calls = self._convert_tool_calls(ember_response.tool_calls)

                # Handle function calls (legacy format)
                elif hasattr(ember_response, "function_call") and ember_response.function_call:
                    message.function_call = ember_response.function_call

                # Create choice
                choice = Choice(index=0, message=message, finish_reason="stop")

                # Extract usage if available
                # ember_model.complete() returns ChatResponse with UsageStats
                usage = None
                if hasattr(ember_response, "usage") and ember_response.usage is not None:
                    usage = Usage(
                        prompt_tokens=ember_response.usage.prompt_tokens,
                        completion_tokens=ember_response.usage.completion_tokens,
                        total_tokens=ember_response.usage.total_tokens,
                    )

                return ChatCompletion(id=completion_id, model=model, choices=[choice], usage=usage)

            def _convert_tool_calls(self, ember_tool_calls: List[Any]) -> List[Dict[str, Any]]:
                """Convert Ember tool calls to OpenAI format."""
                openai_tool_calls = []

                for _index, tc in enumerate(ember_tool_calls):
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": (
                                tc.function.name
                                if hasattr(tc.function, "name")
                                else tc.get("name", "")
                            ),
                            "arguments": (
                                tc.function.arguments
                                if hasattr(tc.function, "arguments")
                                else json.dumps(tc.get("arguments", {}))
                            ),
                        },
                    }
                    openai_tool_calls.append(tool_call)

                return openai_tool_calls

            def _stream_response(
                self, ember_model: Any, params: Dict[str, Any], model_name: str
            ) -> Iterator[StreamChatCompletion]:
                """Stream responses in OpenAI format."""
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

                # First chunk with role
                yield StreamChatCompletion(
                    id=completion_id,
                    model=model_name,
                    choices=[StreamChoice(index=0, delta={"role": "assistant"})],
                )

                # Stream content chunks
                try:
                    for chunk in ember_model.stream(**params):
                        content = ""
                        if hasattr(chunk, "content"):
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        elif isinstance(chunk, dict) and "content" in chunk:
                            content = chunk["content"]

                        if content:
                            yield StreamChatCompletion(
                                id=completion_id,
                                model=model_name,
                                choices=[StreamChoice(index=0, delta={"content": content})],
                            )

                    # Final chunk
                    yield StreamChatCompletion(
                        id=completion_id,
                        model=model_name,
                        choices=[StreamChoice(index=0, delta={}, finish_reason="stop")],
                    )

                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    # Send error in final chunk
                    yield StreamChatCompletion(
                        id=completion_id,
                        model=model_name,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta={"content": f"\n[Error: {str(e)}]"},
                                finish_reason="error",
                            )
                        ],
                    )

    def _get_ember_model(self, model_name: str):
        """Get or create an Ember model instance."""
        if model_name not in self._model_cache:
            try:
                self._model_cache[model_name] = models.get_model(model_name)
                logger.debug(f"Created Ember model instance for: {model_name}")
            except Exception as e:
                logger.error(f"Failed to create model '{model_name}': {e}")
                # Fall back to default model
                if model_name != self.default_model:
                    logger.warning(f"Falling back to default model: {self.default_model}")
                    return self._get_ember_model(self.default_model)
                raise

        return self._model_cache[model_name]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get aggregated usage statistics.

        Returns:
            Dictionary containing usage metrics across all models
        """
        stats = {
            "models_used": list(self._model_cache.keys()),
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        # Aggregate metrics from context
        if hasattr(self.metrics_context, "get_aggregated_metrics"):
            metrics = self.metrics_context.get_aggregated_metrics()
            stats.update(metrics)

        return stats
