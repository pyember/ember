from ember.models.catalog import ModelInfo
from ember.models.discovery.merge import merge_catalog
from ember.models.discovery.types import DiscoveredModel, ModelKey


def test_merge_catalog_layers_dynamic_bootstrap_and_overrides():
    bootstrap = {
        "gpt-4": ModelInfo(
            id="gpt-4",
            provider="openai",
            description="Most capable GPT-4 model",
            context_window=8192,
        )
    }

    dynamic = {
        ModelKey.to_key("google", "gemini-2.5-pro"): DiscoveredModel(
            provider="google",
            id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            capabilities=("generateContent", "embedContent"),
            context_window_in=1_000_000,
        )
    }

    overrides = {
        "google:gemini-2.5-pro": {
            "description": "Vertex contract tier",
            "pricing": {"input_per_million": 0.9},
            "hidden": False,
            "capabilities": ["generateContent", "embedContent", "generateAnswers"],
        }
    }

    merged = merge_catalog(dynamic=dynamic, bootstrap=bootstrap, overrides=overrides)

    key = ModelKey.to_key("google", "gemini-2.5-pro")
    assert key in merged
    google_record = merged[key]
    assert google_record["provider"] == "google"
    assert google_record["id"] == "gemini-2.5-pro"
    assert google_record["description"] == "Vertex contract tier"
    assert google_record["pricing_override"] == {"input_per_million": 0.9}
    assert google_record["hidden"] is False
    assert google_record["capabilities"] == [
        "embedContent",
        "generateAnswers",
        "generateContent",
    ]

    openai_key = ModelKey.to_key("openai", "gpt-4")
    assert openai_key in merged
    openai_record = merged[openai_key]
    assert openai_record["provider"] == "openai"
    assert openai_record["context_window"] == 8192
    assert "gpt-4" in openai_record["aliases"]
