"""Ember language model backend for DSPy."""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    from dspy.primitives.language_model import BaseLM
except ImportError as exc:
    raise ImportError(
        "DSPy is required for this integration. Install with: pip install dspy-ai"
    ) from exc

from ember._internal.context import MetricsContext
from ember.api import models

logger = logging.getLogger(__name__)


class EmberLM(BaseLM):
    """Ember language model backend for DSPy.

    This class provides a bridge between DSPy's language model interface and
    Ember's unified model API, enabling use of any Ember-supported model
    within DSPy programs.

    Args:
        model: Model identifier (e.g., "claude-3-opus-20240229")
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        cache: Whether to cache responses
        **kwargs: Additional model parameters

    Example:
        >>> from ember.integrations.dspy import EmberLM
        >>> import dspy
        >>>
        >>> # Initialize with Ember model
        >>> lm = EmberLM(model="gpt-4", temperature=0.7)
        >>> dspy.configure(lm=lm)
        >>>
        >>> # Use with DSPy modules
        >>> classify = dspy.Predict("text -> sentiment")
        >>> result = classify(text="I love this!")
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        **kwargs,
    ):
        # Initialize parent with DSPy expected parameters
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        # Initialize Ember model
        self.model_id = model
        try:
            self.ember_model = models.get_model(model)
            logger.info("Initialized EmberLM with model: %s", model)
        except Exception as err:  # pragma: no cover - defensive logging
            raise ValueError(f"Failed to initialize Ember model '{model}': {err}") from err

        # Metrics tracking
        self.metrics_context = MetricsContext()

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Execute model call through Ember.

        Args:
            prompt: Text prompt (for completion-style calls)
            messages: List of message dicts (for chat-style calls)
            **kwargs: Additional parameters to override defaults

        Returns:
            Generated text response
        """
        # Merge kwargs with instance defaults
        call_kwargs = {**self.kwargs, **kwargs}

        # Convert inputs to Ember format
        if messages is not None:
            ember_messages = self._convert_messages(messages)
        elif prompt is not None:
            ember_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Extract Ember-specific parameters
        temperature = call_kwargs.pop("temperature", self.temperature)
        max_tokens = call_kwargs.pop("max_tokens", self.max_tokens)

        # Call Ember model with metrics tracking
        with self.metrics_context.track(model=self.model_id) as span:
            try:
                response = self.ember_model.complete(
                    messages=ember_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                span.record_response(response, model=self.model_id)

                # Extract content from response
                if hasattr(response, "content"):
                    content = response.content
                elif isinstance(response, dict) and "content" in response:
                    content = response["content"]
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)

                # Track in DSPy history
                self._track_history(prompt or messages, content, call_kwargs)

                return content

            except Exception as err:  # pragma: no cover - defensive logging
                logger.error("Ember model call failed: %s", err)
                raise RuntimeError(f"Model call failed: {err}") from err

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert DSPy message format to Ember format.

        Args:
            messages: DSPy format messages

        Returns:
            Ember format messages
        """
        ember_messages = []

        for msg in messages:
            # Handle various message formats
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle DSPy's potential format variations
                if not content and "text" in msg:
                    content = msg["text"]
                elif not content and "message" in msg:
                    content = msg["message"]

                ember_messages.append({"role": role, "content": content})
            elif isinstance(msg, str):
                # Plain string messages default to user role
                ember_messages.append({"role": "user", "content": msg})
            else:
                logger.warning(f"Unexpected message format: {type(msg)}")
                ember_messages.append({"role": "user", "content": str(msg)})

        return ember_messages

    def _track_history(self, input_data: Any, output: str, kwargs: Dict[str, Any]):
        """Track call in DSPy history format.

        Args:
            input_data: Original input (prompt or messages)
            output: Generated response
            kwargs: Call parameters
        """
        metrics = self.metrics_context.get_last_metrics()

        history_entry = {
            "prompt": (input_data if isinstance(input_data, str) else json.dumps(input_data)),
            "response": output,
            "kwargs": kwargs,
            "ember_metrics": {
                "model": self.model,
                "usage": metrics.get("usage", {}),
                "latency_ms": metrics.get("latency_ms", 0),
                "cost": metrics.get("cost", {}),
            },
        }

        self.history.append(history_entry)

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get aggregated usage metrics from all calls.

        Returns:
            Dictionary containing usage statistics
        """
        if not self.history:
            return {}

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_calls = len(self.history)

        for entry in self.history:
            metrics = entry.get("ember_metrics", {})
            usage = metrics.get("usage", {})
            cost = metrics.get("cost", {})

            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)
            total_cost += cost.get("total", 0.0)

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": total_cost,
            "average_cost_per_call": total_cost / total_calls if total_calls > 0 else 0,
        }

    def inspect_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Inspect recent history entries.

        Args:
            n: Number of entries to return (None for all)

        Returns:
            List of history entries
        """
        if n is None:
            return self.history
        return self.history[-n:]

    def __repr__(self) -> str:
        """String representation of EmberLM."""
        return (
            "EmberLM("
            f"model='{self.model}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}"
            ")"
        )
