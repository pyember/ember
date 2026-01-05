from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover
    from ember.models.providers.openai._responses import ResponsesHandler


def _project_root() -> Path:
    marker = Path(__file__).resolve()
    while marker.name != "tests" and marker.parent != marker:
        marker = marker.parent
    return marker.parent


def _load_responses_handler() -> "ResponsesHandler":
    root = _project_root()
    src_root = root / "src" / "ember"

    module_names = [
        "ember",
        "ember._internal",
        "ember._internal.exceptions",
        "ember.models",
        "ember.models.schemas",
        "ember.models.providers",
        "ember.models.providers.openai",
        "ember.models.providers.openai._messages",
        "ember.models.providers.openai._types",
        "ember.models.providers.openai._usage",
        "ember.models.providers.openai._responses",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    def _cleanup() -> None:
        for name, original in saved_modules.items():
            if original is not None:
                sys.modules[name] = original
            elif name in sys.modules:
                del sys.modules[name]

    try:
        ember_pkg = ModuleType("ember")
        ember_pkg.__path__ = [str(src_root)]  # type: ignore[attr-defined]
        sys.modules["ember"] = ember_pkg

        def _ensure_namespace(name: str, path: Path) -> None:
            module = sys.modules.get(name)
            if module is None:
                module = ModuleType(name)
                sys.modules[name] = module
            module.__path__ = [str(path)]  # type: ignore[attr-defined]

        _ensure_namespace("ember._internal", src_root / "_internal")
        _ensure_namespace("ember.models", src_root / "models")
        _ensure_namespace("ember.models.schemas", src_root / "models" / "schemas")
        _ensure_namespace("ember.models.providers", src_root / "models" / "providers")
        _ensure_namespace(
            "ember.models.providers.openai",
            src_root / "models" / "providers" / "openai",
        )

        def _load_module(name: str, relative_path: str) -> ModuleType:
            module_path = src_root.parent / relative_path
            spec = util.spec_from_file_location(name, module_path)
            if spec is None or spec.loader is None:  # pragma: no cover
                raise RuntimeError(f"Failed to load module: {module_path}")
            module = util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)  # type: ignore[assignment]
            return module

        _load_module("ember._internal.exceptions", "ember/_internal/exceptions.py")
        _load_module("ember.models.schemas", "ember/models/schemas/__init__.py")
        _load_module(
            "ember.models.providers.openai._messages",
            "ember/models/providers/openai/_messages.py",
        )
        _load_module(
            "ember.models.providers.openai._types",
            "ember/models/providers/openai/_types.py",
        )
        _load_module(
            "ember.models.providers.openai._usage",
            "ember/models/providers/openai/_usage.py",
        )
        responses_module = _load_module(
            "ember.models.providers.openai._responses",
            "ember/models/providers/openai/_responses.py",
        )
        return responses_module.ResponsesHandler
    finally:
        _cleanup()


class _DummyClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=lambda **_: None, stream=lambda **_: None)


@pytest.fixture()
def handler() -> "ResponsesHandler":
    handler_cls = _load_responses_handler()
    return handler_cls(_DummyClient())


def test_build_payload_keeps_temperature_top_level(handler: ResponsesHandler) -> None:
    payload = handler._build_payload(  # type: ignore[attr-defined]
        "gpt-4o-mini",
        [{"role": "user", "content": "Solve 1+1"}],
        reasoning=None,
        text_cfg=None,
        options={"temperature": 0.7, "max_output_tokens": 64},
    )

    assert pytest.approx(payload["temperature"], rel=0, abs=1e-6) == 0.7
    assert payload["max_output_tokens"] == 64
    assert "text" not in payload or "temperature" not in payload["text"]


def test_build_payload_respects_existing_text_section(handler: ResponsesHandler) -> None:
    payload = handler._build_payload(  # type: ignore[attr-defined]
        "gpt-5",
        [{"role": "user", "content": "Hello"}],
        reasoning=None,
        text_cfg={"verbosity": "low"},
        options={"temperature": 0.25},
    )

    assert "temperature" not in payload
    assert payload["text"]["temperature"] == pytest.approx(0.25, rel=0, abs=1e-6)
    assert payload["text"]["verbosity"] == "low"


def test_prepare_payload_preserves_top_level_temperature(handler: ResponsesHandler) -> None:
    prepared = handler._prepare_payload(  # type: ignore[attr-defined]
        {"model": "gpt-4o-mini", "input": [], "temperature": 0.55},
        model="gpt-4o-mini",
    )

    assert prepared["temperature"] == pytest.approx(0.55, rel=0, abs=1e-6)
    assert "text" not in prepared
