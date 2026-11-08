"""Config file errors name the file and line; removal can repair a misspelt key."""

from pathlib import Path

import pytest

from ember import config, context
from ember.errors import ConfigError


@pytest.fixture(autouse=True)
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("EMBER_CONFIG", str(path))
    context.reload()
    yield path
    context.reload()


@pytest.mark.parametrize(
    ("text", "location", "message"),
    [
        ("default_model: x\nprovders: {}\n", "line 2", "unknown keys ['provders']"),
        ("providers:\n  openai:\n    base_url: 3\n", "line 3", "providers.openai.base_url must be"),
        ("providers:\n  a/b:\n    kind: openai\n", "line 2", "contain no '/'"),
        ("providers:\n  local: x\n", "line 2", "providers.local must be a mapping"),
        ("retry:\n  max_attempts: 3\n  max_atempts: 4\n", "line 3", "retry accepts only"),
        ("retry:\n  max_attempts: three\n", "line 2", "retry.max_attempts must be a number"),
        ("\n\nrequire_price: yes please\n", "line 3", "require_price must be true or false"),
        ("max_in_flight_per_provider: 1.5\n", "line 1", "must be an integer"),
        ("default_model: [unclosed\n", "line 2", "not valid YAML"),
    ],
)
def test_errors_name_the_file_and_line(
    config_file: Path, text: str, location: str, message: str
) -> None:
    config_file.write_text(text)
    with pytest.raises(ConfigError) as raised:
        config.load()
    assert str(raised.value).startswith(f"{config_file} {location}: ")
    assert message in str(raised.value)
    assert "\n" not in str(raised.value)


def test_context_errors_from_the_file_name_the_file(config_file: Path) -> None:
    config_file.write_text("providers:\n  local:\n    base_url: http://h/v1\n")
    with pytest.raises(ConfigError, match=f"^{config_file}: provider 'local' needs kind"):
        context.current()


def test_remove_provider_field_removes_a_misspelt_key(config_file: Path) -> None:
    config_file.write_text("providers:\n  openai:\n    api_key: sk-kept-0000\n    api_kye: x\n")
    config.remove_provider_field("openai", "api_kye")
    assert config.load().providers["openai"] == {"api_key": "sk-kept-0000"}
    assert config_file.stat().st_mode & 0o777 == 0o600


def test_removing_the_last_field_removes_the_provider_and_the_section(config_file: Path) -> None:
    config_file.write_text("default_model: openai/gpt-5\nproviders:\n  openai:\n    base_url: u\n")
    config.remove_provider_field("openai", "base_url")
    assert config_file.read_text() == "default_model: openai/gpt-5\n"


def test_remove_refuses_what_is_not_set(config_file: Path) -> None:
    config_file.write_text("providers:\n  openai:\n    base_url: u\n")
    with pytest.raises(ConfigError, match="providers.openai.api_key is not set"):
        config.remove_provider_field("openai", "api_key")
    with pytest.raises(ConfigError, match="providers.gemini is not set"):
        config.remove_provider_field("gemini", None)


def test_remove_does_not_write_a_file_that_stays_invalid(config_file: Path) -> None:
    text = "provders: {}\nproviders:\n  openai:\n    api_kye: x\n"
    config_file.write_text(text)
    with pytest.raises(ConfigError, match="unknown keys \\['provders'\\]"):
        config.remove_provider_field("openai", "api_kye")
    assert config_file.read_text() == text
