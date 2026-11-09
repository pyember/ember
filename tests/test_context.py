import re
import sys
import threading
from pathlib import Path

import pytest

from ember import config, context
from ember.errors import ConfigError


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("EMBER_CONFIG", str(path))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield path
    context.reload()


def test_missing_file_gives_builtin_providers() -> None:
    assert set(context.current().providers) == {"openai", "anthropic", "gemini"}


def test_invalid_file_raises_config_error_naming_the_path(isolated_config: Path) -> None:
    isolated_config.write_text("providers: [unclosed\n")
    with pytest.raises(ConfigError, match=re.escape(str(isolated_config))):
        context.current()


def test_unknown_keys_are_rejected(isolated_config: Path) -> None:
    isolated_config.write_text("provders: {}\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        context.current()


def test_override_merges_provider_fields_and_restores(isolated_config: Path) -> None:
    isolated_config.write_text(
        "providers:\n  local:\n    kind: openai_compatible\n    base_url: http://h/v1\n"
    )
    with context.override(providers={"local": {"api_key": "sk-override-0000"}}) as ctx:
        local = ctx.provider("local")
        assert local.base_url == "http://h/v1"
        assert local.credential() is not None
        assert local.credential().reveal() == "sk-override-0000"
    assert context.current().provider("local").credential() is None


def test_credential_precedence(monkeypatch: pytest.MonkeyPatch, isolated_config: Path) -> None:
    isolated_config.write_text("providers:\n  openai:\n    api_key: sk-from-file-0000\n")
    assert context.current().provider("openai").credential().reveal() == "sk-from-file-0000"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-00000")
    assert context.current().provider("openai").credential().reveal() == "sk-from-env-00000"
    with context.override(providers={"openai": {"api_key": "sk-override-00000"}}):
        assert context.current().provider("openai").credential().reveal() == "sk-override-00000"


def test_overrides_are_isolated_per_thread() -> None:
    seen: dict[str, bool] = {}

    def worker(name: str, value: bool) -> None:
        with context.override(require_price=value):
            barrier.wait()
            seen[name] = context.current().require_price

    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=worker, args=("a", True)),
        threading.Thread(target=worker, args=("b", False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {"a": True, "b": False}


def test_nested_overrides_unwind_in_order() -> None:
    with context.override(max_in_flight_per_provider=4):
        with context.override(max_in_flight_per_provider=2):
            assert context.current().max_in_flight_per_provider == 2
        assert context.current().max_in_flight_per_provider == 4
    assert context.current().max_in_flight_per_provider == 64


def test_secrets_never_appear_in_reprs(isolated_config: Path) -> None:
    isolated_config.write_text("providers:\n  openai:\n    api_key: sk-secret-abcd1234\n")
    settings = context.current().provider("openai")
    assert "sk-secret" not in repr(settings)
    assert config.redacted(config.load())["providers"]["openai"]["api_key"] == "…1234"


def test_save_provider_field_writes_atomically_with_private_mode(isolated_config: Path) -> None:
    config.save_provider_field("openai", "api_key", "sk-saved-00000000")
    if sys.platform != "win32":  # Windows has no mode bits.
        assert isolated_config.stat().st_mode & 0o777 == 0o600
    context.reload()
    assert context.current().provider("openai").credential().reveal() == "sk-saved-00000000"
