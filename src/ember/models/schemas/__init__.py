"""Typed schemas shared across the Ember models stack.

All provider configuration, model metadata, request/response payloads, and
usage tracking primitives live here to keep the public API consistent and
well-typed. The dataclasses intentionally avoid side effects so they stay easy
to serialize and reason about.

Examples:
    >>> from ember.models.schemas import ChatRequest, ChatResponse
    >>> request = ChatRequest(prompt='ping', max_tokens=16)
    >>> response = ChatResponse(data='pong')
    >>> response.data
    'pong'

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Dict, List, Mapping, Optional, Protocol, TypedDict, Union, runtime_checkable


@runtime_checkable
class StreamEvent(Protocol):
    """Protocol for streaming events emitted by providers."""

    type: str


def _aggregate_output_text(output: List[Dict[str, Any]]) -> str:
    """Concatenate text-like content from Responses output payloads."""

    chunks: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content_list = item.get("content")
        if isinstance(content_list, list):
            for part in content_list:
                if isinstance(part, dict):
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
                elif isinstance(part, str):
                    chunks.append(part)
        elif isinstance(content_list, str):
            chunks.append(content_list)
        text_val = item.get("text")
        if isinstance(text_val, str):
            chunks.append(text_val)
    return "".join(chunks)


def _merge_usage_details(
    base: Optional[Mapping[str, Any]],
    incoming: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Combine two usage metadata dictionaries."""

    if not base and not incoming:
        return None

    merged: Dict[str, Any] = {}
    if base:
        merged.update(base)
    if incoming:
        for key, value in incoming.items():
            if key in merged:
                existing = merged[key]
                if isinstance(existing, (int, float)) and isinstance(value, (int, float)):
                    merged[key] = existing + value
                elif existing == value:
                    continue
                else:
                    merged[key] = value
            else:
                merged[key] = value
    return merged


# Provider-related schemas


@dataclass
class ProviderInfo:
    """Describe how to authenticate and reach a model provider.

    Attributes:
        name: Canonical provider identifier such as 'openai' or 'google'.
        default_api_key: API key used when a request does not supply one explicitly.
        base_url: Alternate API endpoint for private deployments or proxies.
        custom_args: Free-form provider tuning knobs that callers may supply.

    Examples:
        >>> ProviderInfo(name='anthropic').base_url is None
        True
    """

    name: str
    default_api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_args: Optional[Dict[str, Any]] = None


class ProviderParams(TypedDict, total=False):
    """Provider-specific overrides that flow directly to SDK calls.

    All keys are optional so that callers can adopt provider extensions
    incrementally without breaking compatibility across providers.

    Examples:
        >>> ProviderParams(system='You are friendly', stop=['END'])
        {'system': 'You are friendly', 'stop': ['END']}
    """

    # OpenAI specific
    response_format: Optional[Dict[str, str]]
    seed: Optional[int]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]

    # Anthropic specific
    system: Optional[str]
    metadata: Optional[Dict[str, Any]]

    # Common
    stop: Optional[Union[str, List[str]]]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    user: Optional[str]
    max_output_tokens: Optional[int]
    reasoning: Optional[Dict[str, Any]]
    text: Optional[Dict[str, Any]]


class TextVerbosity(StrEnum):
    """Text output verbosity controls exposed by GPT-5."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TextConfig:
    """Auxiliary text generation controls exposed by GPT-5."""

    verbosity: TextVerbosity

    def to_openai_payload(self) -> Dict[str, str]:
        return {"verbosity": self.verbosity.value}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TextConfig":
        verbosity = data.get("verbosity")
        if not isinstance(verbosity, str):
            raise ValueError("text.verbosity must be provided as a string")
        return cls(verbosity=TextVerbosity(verbosity.lower()))


def normalize_text_config(value: Optional[TextConfig | Mapping[str, Any]]) -> Optional[TextConfig]:
    if value is None:
        return None
    if isinstance(value, TextConfig):
        return value
    if isinstance(value, Mapping):
        return TextConfig.from_mapping(value)
    raise TypeError("text must be a mapping or TextConfig instance")


class ReasoningEffort(StrEnum):
    """Reasoning effort levels for models with extended thinking."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ReasoningConfig:
    """Reasoning controls for models with extended thinking capabilities."""

    effort: ReasoningEffort

    def to_openai_payload(self) -> Dict[str, str]:
        return {"effort": self.effort.value}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReasoningConfig":
        effort = data.get("effort")
        if not isinstance(effort, str):
            raise ValueError("reasoning.effort must be provided as a string")
        return cls(effort=ReasoningEffort(effort.lower()))


def normalize_reasoning_config(
    value: Optional[ReasoningConfig | Mapping[str, Any]],
) -> Optional[ReasoningConfig]:
    if value is None:
        return None
    if isinstance(value, ReasoningConfig):
        return value
    if isinstance(value, Mapping):
        return ReasoningConfig.from_mapping(value)
    raise TypeError("reasoning must be a mapping or ReasoningConfig instance")


# Model configuration schemas


@dataclass
class ModelCost:
    """Token pricing for a single model.

    Attributes:
        input_cost_per_million: USD price per 1,000,000 prompt tokens.
        output_cost_per_million: USD price per 1,000,000 completion tokens.

    Examples:
        >>> cost = ModelCost(input_cost_per_million=15.0, output_cost_per_million=60.0)
        >>> round(cost.output_cost_per_token, 6)
        6e-05
    """

    input_cost_per_million: float
    output_cost_per_million: float

    @property
    def input_cost_per_token(self) -> float:
        """Return the USD price for a single input token.

        Returns:
            float: Cost of one prompt token.
        """
        return self.input_cost_per_million / 1_000_000.0

    @property
    def output_cost_per_token(self) -> float:
        """Return the USD price for a single output token.

        Returns:
            float: Cost of one completion token.
        """
        return self.output_cost_per_million / 1_000_000.0


@dataclass
class RateLimit:
    """Describe throughput limits enforced by a provider.

    Attributes:
        tokens_per_minute: Maximum combined prompt and completion tokens per minute.
        requests_per_minute: Maximum API calls allowed per minute.

    Examples:
        >>> RateLimit(tokens_per_minute=50000).tokens_per_minute
        50000
    """

    tokens_per_minute: Optional[int] = None
    requests_per_minute: Optional[int] = None


@dataclass
class ModelInfo:
    """Aggregate metadata about a registered model.

    Attributes:
        id: Canonical identifier used throughout Ember.
        name: Human-readable label used in UIs and logs.
        provider: Provider slug that serves the model.
        context_window: Maximum supported tokens for prompts plus completions.
        cost: Optional pricing information for cost estimation.
        rate_limit: Optional rate limiting policy for the model.
        api_key: Optional credential override applied to this model only.
        base_url: Optional endpoint override applied to this model only.
        supports_streaming: True when the provider can stream responses.
        supports_functions: True when tool/function calling is available.
        supports_vision: True when the model accepts multimodal inputs.

    Examples:
        >>> ModelInfo(id="gpt-4", name="GPT-4", provider="openai").supports_streaming
        False
    """

    id: str
    name: str
    provider: str
    context_window: int = 4096
    cost: Optional[ModelCost] = None
    rate_limit: Optional[RateLimit] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_vision: bool = False

    def get_api_key(self, provider_info: Optional[ProviderInfo] = None) -> Optional[str]:
        """Resolve the credential to use for this model.

        Args:
            provider_info: Provider metadata supplying a fallback API key.

        Returns:
            Optional[str]: Credential selected for the request.

        Examples:
            >>> model = ModelInfo(id='x', name='X', provider='openai', api_key='sk-test')
            >>> model.get_api_key()
            'sk-test'
        """
        if self.api_key:
            return self.api_key
        if provider_info and provider_info.default_api_key:
            return provider_info.default_api_key
        return None

    def get_base_url(self, provider_info: Optional[ProviderInfo] = None) -> Optional[str]:
        """Return the API endpoint to call for this model.

        Args:
            provider_info: Provider metadata supplying a default endpoint.

        Returns:
            Optional[str]: Base URL that should be used for requests.

        Examples:
            >>> ModelInfo(id='x', name='X', provider='openai').get_base_url() is None
            True
        """
        if self.base_url:
            return self.base_url
        if provider_info and provider_info.base_url:
            return provider_info.base_url
        return None


# Request/Response schemas


@dataclass
class ChatRequest:
    """Provider-agnostic request payload for chat completions.

    Attributes:
        prompt: Primary instruction text sent to the model.
        context: Optional chat history in provider-neutral form.
        max_tokens: Maximum number of completion tokens to generate.
        temperature: Sampling temperature where 0 is deterministic.
        top_p: Nucleus sampling probability mass cutoff.
        stop: Optional stop sequences terminating generation early.
        stream: True to request incremental streaming responses.
        provider_params: Provider-specific keyword arguments.

    Examples:
        >>> ChatRequest(prompt="Hello", temperature=0.2).stream
        False
    """

    prompt: str
    context: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    provider_params: Optional[ProviderParams] = None

    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert this request into provider invocation keyword arguments.

        The returned mapping is designed to be forwarded as ``**kwargs`` into
        Ember provider adapters (``BaseProvider.complete`` / ``stream_complete``).
        It intentionally omits the model identifier because Ember routes model
        selection separately.

        Args:
            provider: Provider slug requesting the conversion.

        Examples:
            >>> ChatRequest(prompt="hi", max_tokens=5).to_provider_format("openai")["max_tokens"]
            5
        """
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")

        payload: Dict[str, Any] = {}
        if self.provider_params is not None:
            payload.update(dict(self.provider_params))

        if self.context is not None:
            payload["context"] = self.context
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.stream:
            payload["stream"] = True

        return payload


@dataclass
class ChatResponse:
    """Normalized representation of a provider response.

    Attributes:
        data: Generated text returned by the provider.
        usage: Optional usage and cost metrics.
        model_id: Identifier of the model producing the response.
        raw_output: Vendor-specific payload retained for debugging.
        thinking_trace: Extended thinking/reasoning trace when available.
        created_at: UTC timestamp captured when the response was created.
        latency_ms: Wall-clock latency recorded by the registry (milliseconds).
        started_at: UTC timestamp captured immediately before invoking the provider.
        completed_at: UTC timestamp captured after the provider returns.

    Examples:
        >>> ChatResponse(data='ok', model_id='gpt-4').model_id
        'gpt-4'
    """

    data: str
    output: Optional[List[Dict[str, Any]]] = None
    response_id: Optional[str] = None
    usage: Optional[UsageStats] = None
    model_id: Optional[str] = None
    raw_output: Optional[Any] = None
    thinking_trace: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def text(self) -> str:
        """Return the generated text payload.

        Returns:
            str: Convenience alias for ``data``.

        Examples:
            >>> ChatResponse(data='hi').text
            'hi'
        """
        if self.data:
            return self.data
        if self.output:
            return _aggregate_output_text(self.output)
        return ""


@dataclass
class EmbeddingResponse:
    """Normalized embedding payload returned by providers.

    Attributes:
        embeddings: Embedding vectors ordered to align with requested inputs.
        usage: Optional token usage/cost statistics.
        model_id: Identifier of the embedding model.
        raw_output: Raw provider response retained for debugging.
        created_at: UTC timestamp captured on creation.
        latency_ms: Wall-clock latency for the embedding call.
        started_at: UTC timestamp before invoking the provider.
        completed_at: UTC timestamp when the provider responded.
    """

    embeddings: List[List[float]]
    usage: Optional[UsageStats] = None
    model_id: Optional[str] = None
    raw_output: Optional[Any] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Usage tracking schemas


@dataclass
class UsageStats:
    """Capture token usage and optional cost estimates.

    Attributes:
        prompt_tokens: Number of prompt tokens consumed.
        completion_tokens: Number of completion tokens generated.
        total_tokens: Aggregate tokens (auto-filled when omitted).
        cost_usd: Estimated USD cost calculated from pricing tables.
        actual_cost_usd: Provider-reported USD cost when available.

    Examples:
        >>> UsageStats(prompt_tokens=10, completion_tokens=5).total_tokens
        15
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Synchronize total_tokens with prompt and completion counts.

        Returns:
            None
        """
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: "UsageStats") -> None:
        """Accumulate another usage sample into this instance.

        Args:
            other: Usage statistics to merge.

        Returns:
            None

        Examples:
            >>> total = UsageStats()
            >>> total.add(UsageStats(prompt_tokens=2, completion_tokens=3))
            >>> total.total_tokens
            5
        """
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        if self.cost_usd is not None and other.cost_usd is not None:
            self.cost_usd += other.cost_usd
        elif other.cost_usd is not None:
            self.cost_usd = other.cost_usd
        self.details = _merge_usage_details(self.details, other.details)

    def __add__(self, other: UsageStats) -> UsageStats:
        """Return a new instance combining two usage samples.

        Args:
            other: Usage statistics to merge.

        Returns:
            UsageStats: Aggregated usage metrics.

        Examples:
            >>> combined = UsageStats(prompt_tokens=1) + UsageStats(completion_tokens=2)
            >>> combined.total_tokens
            3
        """
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=(self.cost_usd or 0) + (other.cost_usd or 0),
            details=_merge_usage_details(self.details, other.details),
        )


@dataclass
class UsageRecord:
    """Immutable snapshot of a single model invocation.

    Attributes:
        usage: Usage statistics collected for the invocation.
        timestamp: UTC timestamp captured when the record was created.
        model_id: Identifier of the model that produced the usage.
        request_id: Optional correlation identifier for traceability.

    Examples:
        >>> UsageRecord(usage=UsageStats(prompt_tokens=1)).model_id is None
        True
    """

    usage: UsageStats
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    model_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class UsageSummary:
    """Aggregate usage metrics for a single model.

    Attributes:
        model_name: Identifier associated with the tracked usage.
        total_usage: Cumulative usage metrics aggregated so far.
        request_count: Number of recorded invocations.
        first_used: Timestamp of the first recorded invocation.
        last_used: Timestamp of the most recent invocation.

    Examples:
        >>> summary = UsageSummary(model_name='gpt-4')
        >>> summary.add_usage(UsageStats(prompt_tokens=1, completion_tokens=2))
        >>> summary.request_count
        1
    """

    model_name: str
    total_usage: UsageStats = field(default_factory=UsageStats)
    request_count: int = 0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None

    def add_usage(self, usage: UsageStats) -> None:
        """Include a usage sample in the running summary.

        Args:
            usage: Usage metrics from a single invocation.

        Returns:
            None

        Examples:
            >>> summary = UsageSummary(model_name='gpt-4')
            >>> summary.add_usage(UsageStats(prompt_tokens=3, completion_tokens=1))
            >>> summary.total_usage.total_tokens
            4
        """
        self.total_usage.add(usage)
        self.request_count += 1

        now = datetime.now(UTC)
        if self.first_used is None:
            self.first_used = now
        self.last_used = now


# Type aliases for clarity and better type hints throughout the codebase.
# These make function signatures more readable and self-documenting.
ModelID = str  # e.g., "gpt-4", "claude-3-opus"
ProviderName = str  # e.g., "openai", "anthropic"
ResponseData = Union[str, Dict[str, Any]]  # Flexible response format
