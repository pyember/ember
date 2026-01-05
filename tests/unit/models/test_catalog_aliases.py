import sys
import types


def _install_anthropic_stub() -> None:
    """Provide a minimal anthropic stub so catalog import does not require the SDK."""

    if "anthropic" in sys.modules:
        return

    anthropic_module = types.ModuleType("anthropic")
    anthropic_types = types.ModuleType("anthropic.types")

    class _StubAPIError(Exception):
        pass

    anthropic_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
    anthropic_module.RateLimitError = type("RateLimitError", (Exception,), {})
    anthropic_module.APIError = type("APIError", (_StubAPIError,), {})
    anthropic_module.__version__ = "0"

    anthropic_types.MessageParam = object
    anthropic_types.TextBlock = type("TextBlock", (), {"text": ""})
    anthropic_types.ThinkingBlock = type("ThinkingBlock", (), {"thinking": ""})
    anthropic_types.RedactedThinkingBlock = type("RedactedThinkingBlock", (), {"data": ""})
    anthropic_types.ToolUseBlock = type("ToolUseBlock", (), {})
    anthropic_types.ServerToolUseBlock = type("ServerToolUseBlock", (), {})
    anthropic_types.WebSearchToolResultBlock = type("WebSearchToolResultBlock", (), {})
    anthropic_module.types = anthropic_types

    anthropic_module.Anthropic = type("Anthropic", (), {})

    sys.modules["anthropic"] = anthropic_module
    sys.modules["anthropic.types"] = anthropic_types


_install_anthropic_stub()

from ember.models import catalog  # noqa: E402  (requires stub before import)


class _StubDynamicCatalog:
    def __init__(self, payload):
        self._payload = payload

    def load(self, **kwargs):
        return dict(self._payload)

    def clear_cache(self) -> None:  # pragma: no cover - compatibility
        self._payload = {}


def test_canonicalize_uses_dynamic_alias(monkeypatch):
    dynamic_payload = {
        "stub:alpha": {
            "provider": "stub",
            "id": "alpha",
            "aliases": ["alpha"],
            "description": "Alpha model",
            "context_window": 1024,
        }
    }

    monkeypatch.setattr(catalog, "_DYNAMIC_CATALOG", _StubDynamicCatalog(dynamic_payload))
    monkeypatch.setattr(catalog, "_resolve_overrides", lambda: {})

    canonical, defaults = catalog.canonicalize_model_identifier("alpha")

    assert canonical == "alpha"
    assert defaults == {}


def test_canonicalize_applies_override_alias(monkeypatch):
    dynamic_payload = {
        "stub:alpha": {
            "provider": "stub",
            "id": "alpha",
            "aliases": ["alpha"],
            "description": "Alpha model",
            "context_window": 1024,
        }
    }
    overrides = {"stub:alpha": {"aliases": ["alt-alpha"]}}

    monkeypatch.setattr(catalog, "_DYNAMIC_CATALOG", _StubDynamicCatalog(dynamic_payload))
    monkeypatch.setattr(catalog, "_resolve_overrides", lambda: overrides)

    canonical, defaults = catalog.canonicalize_model_identifier("alt-alpha")

    assert canonical == "alpha"
    assert defaults == {}


def test_canonicalize_preserves_gpt5_reasoning_defaults():
    canonical, defaults = catalog.canonicalize_model_identifier("gpt-5-high")

    assert canonical == "gpt-5"
    assert defaults == {"reasoning": {"effort": "high"}}


def test_canonicalize_supports_gpt5_xhigh_reasoning_defaults():
    canonical, defaults = catalog.canonicalize_model_identifier("gpt-5-xhigh")

    assert canonical == "gpt-5"
    assert defaults == {"reasoning": {"effort": "xhigh"}}
