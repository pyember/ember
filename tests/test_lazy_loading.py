import sys


def test_ember_import_is_lightweight():
    # Ensure heavy modules are NOT in sys.modules
    heavy_modules = ["anthropic", "google.generativeai", "openai"]
    for mod in heavy_modules:
        if mod in sys.modules:
            del sys.modules[mod]
            
    import importlib
    providers = importlib.import_module("ember.models.providers")
    
    # Check that heavy modules are still not imported
    for mod in heavy_modules:
        assert mod not in sys.modules, f"{mod} was imported unexpectedly!"

    # But we can resolve models
    provider, _ = providers.resolve_model_id("gpt-4")
    assert provider == "openai"
    
    # And getting the class DOES import
    cls = providers.get_provider_class("openai")
    assert cls.__name__ == "OpenAIProvider"
    assert "ember.models.providers.openai" in sys.modules
