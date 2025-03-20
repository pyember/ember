"""This module is a placeholder example for observability metrics. We are excited to expand on this in the future."""
from prometheus_client import CollectorRegistry, Counter, Histogram


def create_metrics() -> dict[str, object]:
    """Creates and returns the isolated metrics objects with a custom registry."""
    # Create a custom registry to avoid clashing with the global one.
    registry = CollectorRegistry()

    metrics = {
        "model_invocations": Counter(
            name="model_invocations_total",
            documentation="Total invocations",
            labelnames=["model_id"],
            registry=registry,
        ),
        "invocation_duration": Histogram(
            name="model_invocation_duration_seconds",
            documentation="Invocation duration",
            labelnames=["model_id"],
            registry=registry,
        ),
        "registry": registry,
    }
    return metrics


# Singleton-like metrics instance to be imported wherever needed.
METRICS: dict[str, object] = create_metrics()
