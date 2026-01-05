import importlib

from ember.models.catalog import MODEL_CATALOG, Models


def _sanitize_constant_name(model_id: str) -> str:
    out = []
    for ch in model_id:
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    name = "".join(out)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def test_models_constants_match_stable_catalog():
    expected = {
        _sanitize_constant_name(mid): mid
        for mid, info in MODEL_CATALOG.items()
        if getattr(info, "status", "stable") == "stable"
    }

    # Validate internal mapping first (no public API change)
    assert hasattr(Models, "_MAP")
    assert Models._MAP == expected

    # Validate attribute access
    for const, mid in expected.items():
        assert getattr(Models, const) == mid


def test_models_constants_are_read_only():
    # Attempt to mutate should raise
    try:
        Models.GPT_4 = "something-else"
        raised = False
    except AttributeError:
        raised = True
    assert raised, "Models should be read-only"

    # Reload module should regenerate mapping without error
    importlib.reload(importlib.import_module("ember.models.catalog"))
