from ember.models.providers import resolve_model_id


def test_plain_ollama_maps_to_auto():
    provider, model = resolve_model_id("ollama")
    assert provider == "ollama"
    assert model in {"auto", "default"}  # default is auto unless overridden by env
