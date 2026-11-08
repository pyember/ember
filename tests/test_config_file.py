"""Config file errors name the file and line; removal can repair a misspelt key."""

import os
import stat
import traceback
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
        ("retry:\n  max_attempts: three\n", "line 2", "retry.max_attempts must be a finite number"),
        ("\n\nrequire_price: yes please\n", "line 3", "require_price must be true or false"),
        ("max_in_flight_per_provider: 1.5\n", "line 1", "must be an integer"),
        ("default_model: [unclosed\n", "line 2", "not valid YAML"),
        ("retry:\n  base_delay_s: .nan\n", "line 2", "retry.base_delay_s must be a finite number"),
        ("retry:\n  max_attempts: .inf\n", "line 2", "retry.max_attempts must be a finite number"),
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


def test_a_yaml_error_does_not_chain_the_parser_error_that_quotes_the_line(
    config_file: Path,
) -> None:
    config_file.write_text('providers:\n  openai:\n    api_key: "sk-secret-abcd1234\n')
    with pytest.raises(ConfigError) as raised:
        config.load()
    rendered = "".join(traceback.format_exception(raised.value))
    assert "sk-secret" not in rendered
    assert f"{config_file} line" in rendered


def test_context_errors_name_the_file_wherever_the_data_is_checked(config_file: Path) -> None:
    config_file.write_text("providers:\n  local:\n    kind: bogus\n")
    data = config.load()
    with pytest.raises(ConfigError, match=f"^{config_file}: provider 'local' has unknown kind"):
        context.from_config(data)


def test_redacted_masks_credentials_inside_urls(config_file: Path) -> None:
    config_file.write_text(
        "providers:\n"
        "  proxy:\n"
        "    kind: openai_compatible\n"
        "    base_url: https://user:sk-secret-abcd1234@proxy.example/v1?key=sk-other-5678#x\n"
        "  plain:\n"
        "    kind: openai_compatible\n"
        "    base_url: http://localhost:11434/v1\n"
    )
    shown = config.redacted(config.load())["providers"]
    assert shown["proxy"]["base_url"] == "https://…@proxy.example/v1?…#x"
    assert shown["plain"]["base_url"] == "http://localhost:11434/v1"


def test_writes_keep_the_file_order(config_file: Path) -> None:
    config_file.write_text("retry:\n  max_attempts: 2\nproviders:\n  openai:\n    base_url: u\n")
    config.save_provider_field("openai", "api_key_env", "MY_KEY")
    assert config_file.read_text() == (
        "retry:\n  max_attempts: 2\nproviders:\n  openai:\n    base_url: u\n"
        "    api_key_env: MY_KEY\n"
    )


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("# my settings\nproviders: {}\n", 1),
        ("providers:\n  openai:\n    base_url: u  # proxy\n", 3),
    ],
)
def test_writers_refuse_to_drop_comments(config_file: Path, text: str, line: int) -> None:
    config_file.write_text(text)
    with pytest.raises(ConfigError, match=f"^{config_file} line {line}: .*comments"):
        config.save_provider_field("openai", "api_key_env", "MY_KEY")
    with pytest.raises(ConfigError, match="comments"):
        config.remove_provider_field("openai", None)
    assert config_file.read_text() == text


def test_a_hash_inside_a_value_is_not_a_comment(config_file: Path) -> None:
    config_file.write_text('providers:\n  openai:\n    base_url: "http://h/v1#frag"\n')
    config.save_provider_field("openai", "api_key_env", "MY_KEY")
    assert config.load().providers["openai"]["base_url"] == "http://h/v1#frag"


def test_writes_are_synced_to_disk(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    config.save_provider_field("openai", "api_key_env", "MY_KEY")
    assert synced == ["file", "dir"]
