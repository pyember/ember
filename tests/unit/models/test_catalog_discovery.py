import importlib

import pytest

from ember.models import catalog
from ember.models.pricing.manager import Pricing


class _DummyDynamicCatalog:
    def __init__(self, payload):
        self._payload = payload

    def load(self, **kwargs):  # noqa: D401 - matches DynamicCatalog signature loosely
        if kwargs.get("mode") == "bootstrap":
            return {}
        overrides = kwargs.get("overrides", {}) or {}
        merged = {k: dict(v) for k, v in self._payload.items()}
        for key, override in overrides.items():
            if key not in merged:
                try:
                    provider, vendor_id = key.split(":", 1)
                except ValueError:
                    provider, vendor_id = "unknown", key
                merged[key] = {
                    "provider": provider,
                    "id": vendor_id,
                    "aliases": [vendor_id],
                }
            record = merged[key]
            if "description" in override:
                record["description"] = override["description"]
            if "pricing" in override:
                record["pricing_override"] = override["pricing"]
            if "hidden" in override:
                record["hidden"] = override["hidden"]
            if "capabilities" in override:
                record["capabilities"] = list(override["capabilities"])
            if "aliases" in override:
                record["aliases"] = list(override["aliases"])
            if "context_window" in override:
                record["context_window"] = override["context_window"]
            if "context_window_out" in override:
                record["context_window_out"] = override["context_window_out"]
        return merged

    def clear_cache(self) -> None:  # pragma: no cover - compatibility stub
        self._payload = {}


def test_list_available_models_includes_dynamic(monkeypatch):
    dynamic_payload = {
        "dummy:demo": {
            "provider": "dummy",
            "id": "demo",
            "aliases": ["demo"],
            "description": "Dynamic demo",
            "context_window": 4096,
            "capabilities": ["stream"],
        }
    }

    monkeypatch.setattr(catalog, "_DYNAMIC_CATALOG", _DummyDynamicCatalog(dynamic_payload))
    monkeypatch.setattr(catalog, "_resolve_overrides", lambda: {})

    models = catalog.list_available_models()
    assert "demo" in models


def test_list_available_models_skip_dynamic_when_disabled(monkeypatch):
    dynamic_payload = {
        "dummy:demo": {
            "provider": "dummy",
            "id": "demo",
            "aliases": ["demo"],
            "description": "Dynamic demo",
            "context_window": 4096,
        }
    }

    monkeypatch.setattr(catalog, "_DYNAMIC_CATALOG", _DummyDynamicCatalog(dynamic_payload))
    monkeypatch.setattr(catalog, "_resolve_overrides", lambda: {})

    models = catalog.list_available_models(include_dynamic=False)
    assert "demo" not in models


def test_get_model_info_applies_overrides(monkeypatch):
    dynamic_payload = {
        "dummy:demo": {
            "provider": "dummy",
            "id": "demo",
            "aliases": ["demo"],
            "description": "Dynamic demo",
            "context_window": 4096,
            "capabilities": ["stream"],
        }
    }
    overrides = {
        "dummy:demo": {
            "description": "Contract tier",
            "pricing": {"input": 1.0, "output": 2.0},
            "hidden": False,
        }
    }

    monkeypatch.setattr(catalog, "_DYNAMIC_CATALOG", _DummyDynamicCatalog(dynamic_payload))
    monkeypatch.setattr(catalog, "_resolve_overrides", lambda: overrides)

    info = catalog.get_model_info("demo", discovery_mode="live")
    assert info.description == "Contract tier"
    assert info.capabilities == ("stream",)
    assert info.pricing_override == {"input": 1.0, "output": 2.0}
    assert info.hidden is False


def test_pricing_manager_respects_overrides(monkeypatch):

    pricing_manager = importlib.import_module("ember.models.pricing.manager")

    monkeypatch.setattr(
        pricing_manager,
        "get_model_overrides",
        lambda: {
            "openai:gpt-4": {"pricing": {"input_per_million": 1.23, "output_per_million": 4.56}}
        },
    )

    pricing = Pricing()
    table = pricing.get_all_pricing()
    assert pytest.approx(table["gpt-4"]["input"], rel=1e-6) == 1.23
    assert pytest.approx(table["gpt-4"]["output"], rel=1e-6) == 4.56
