import importlib.util
import sys
import types
from pathlib import Path


def _load_catalog_module():
    root = Path(__file__).resolve().parents[3]
    src_root = root / "src"
    ember_root = src_root / "ember"

    if "ember" not in sys.modules:
        ember_pkg = types.ModuleType("ember")
        ember_pkg.__path__ = [str(ember_root)]
        sys.modules["ember"] = ember_pkg
    if "ember.models" not in sys.modules:
        models_pkg = types.ModuleType("ember.models")
        models_pkg.__path__ = [str(ember_root / "models")]
        sys.modules["ember.models"] = models_pkg

    path = ember_root / "models" / "catalog" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "ember.models.catalog", str(path), submodule_search_locations=[str(path.parent)]
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_preview_models_not_in_stable_constants():
    mod = _load_catalog_module()
    # Any model marked preview/experimental should not appear in Models._MAP
    stable_map = getattr(mod.Models, "_MAP", {})
    for mid, info in mod.MODEL_CATALOG.items():
        if getattr(info, "status", "stable") != "stable":
            const = "".join(ch.upper() if ch.isalnum() else "_" for ch in mid).strip("_")
            assert const not in stable_map, f"preview model leaked into constants: {mid}"
