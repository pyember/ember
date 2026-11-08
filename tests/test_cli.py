"""The ember command: parses first, never prints or accepts keys, repairs typos."""

import getpass
import subprocess
import sys
from pathlib import Path

import pytest

from ember import context
from ember.cli import main

KEY = "sk-secret-abcdefgh1234"


@pytest.fixture(autouse=True)
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EMBER_CONFIG", str(path))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield path
    context.reload()


@pytest.fixture
def no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(prompt: str = "") -> str:
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(getpass, "getpass", refuse)


def type_key(monkeypatch: pytest.MonkeyPatch, key: str) -> list[str]:
    prompts: list[str] = []

    def answer(prompt: str = "") -> str:
        prompts.append(prompt)
        return key

    monkeypatch.setattr(getpass, "getpass", answer)
    return prompts


def test_a_broken_config_does_not_stop_argument_parsing(config_file: Path) -> None:
    # v0.1 loaded the config before argparse ran, so every command failed.
    config_file.write_text("providers:\n  openai:\n    api_key: x\n   base_url: y\n")
    with pytest.raises(SystemExit) as exited:
        main(["--version"])
    assert exited.value.code == 0


def test_a_broken_config_is_one_line_naming_file_and_line(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text("providers:\n  openai:\n    api_key: x\n   base_url: y\n")
    assert main(["configure", "show"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.startswith(f"ember: error: {config_file} line 4: not valid YAML")


def test_a_misspelt_key_is_located_and_unset_repairs_it(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text(f"providers:\n  openai:\n    api_key: {KEY}\n    api_kye: oops\n")
    assert main(["configure", "show"]) == 1
    assert f"{config_file} line 4: providers.openai: unknown keys ['api_kye']" in (
        capsys.readouterr().err
    )
    assert main(["configure", "unset", "openai", "api_kye"]) == 0
    assert main(["configure", "show"]) == 0
    assert "api_kye" not in config_file.read_text()
    assert KEY in config_file.read_text()


def test_show_never_prints_a_key(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # v0.1's `configure list` and `context view` printed keys in plain text.
    config_file.write_text(f"providers:\n  openai:\n    api_key: {KEY}\n")
    assert main(["configure", "show"]) == 0
    out = capsys.readouterr().out
    assert KEY not in out
    assert KEY[:-4] not in out
    assert "api_key: …1234" in out


def test_keys_are_refused_on_the_command_line(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # v0.1's README told users to run `ember configure set providers.openai.api_key "sk-..."`.
    with pytest.raises(SystemExit) as exited:
        main(["configure", "set", "openai", "api_key", KEY])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "ember configure set-key" in err
    assert KEY not in err
    assert not config_file.exists()


def test_set_key_reads_the_key_at_a_hidden_prompt(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = type_key(monkeypatch, f"  {KEY}\n")
    assert main(["configure", "set-key", "openai"]) == 0
    assert prompts == ["API key for openai (input hidden): "]
    assert KEY not in capsys.readouterr().out
    assert config_file.stat().st_mode & 0o777 == 0o600
    context.reload()
    assert context.current().provider("openai").credential().reveal() == KEY


def test_set_key_says_when_the_environment_wins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-000000")
    type_key(monkeypatch, KEY)
    assert main(["configure", "set-key", "openai"]) == 0
    assert "OPENAI_API_KEY is set and takes precedence" in capsys.readouterr().out


@pytest.mark.usefixtures("no_prompt")
def test_set_key_for_an_unknown_provider_fails_before_prompting(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["configure", "set-key", "opnai"]) == 1
    assert "unknown provider 'opnai'" in capsys.readouterr().err
    assert not config_file.exists()


def test_an_empty_key_saves_nothing(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    type_key(monkeypatch, "   ")
    assert main(["configure", "set-key", "openai"]) == 1
    assert not config_file.exists()


def test_interrupting_the_prompt_exits_130(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(getpass, "getpass", interrupt)
    assert main(["configure", "set-key", "openai"]) == 130
    assert not config_file.exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["set", "local", "base_url", "http://h/v1"], "provider 'local' needs kind"),
        (["set", "local", "kind", "bogus"], "provider 'local' has unknown kind 'bogus'"),
        (["set", "openai", "kind", "anthropic"], "its kind cannot change"),
    ],
)
def test_changes_the_library_would_reject_are_not_written(
    config_file: Path, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    assert main(["configure", *argv]) == 1
    err = capsys.readouterr().err
    assert "nothing was saved" in err and message in err
    assert not config_file.exists()


def test_an_openai_compatible_provider_is_set_up_field_by_field(config_file: Path) -> None:
    assert main(["configure", "set", "local", "kind", "openai_compatible"]) == 0
    assert main(["configure", "set", "local", "base_url", "http://localhost:11434/v1"]) == 0
    context.reload()
    local = context.current().provider("local")
    assert (local.kind, local.base_url) == ("openai_compatible", "http://localhost:11434/v1")


def test_unset_reports_a_config_that_still_does_not_load(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text("providers:\n  local:\n    kind: openai_compatible\n    base_url: x\n")
    assert main(["configure", "unset", "local", "kind"]) == 1
    assert "the config still does not load: provider 'local' needs kind" in (
        capsys.readouterr().err
    )
    assert main(["configure", "unset", "local"]) == 0
    assert "providers" not in config_file.read_text()


def test_unset_of_something_not_set_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["configure", "unset", "openai", "base_url"]) == 1
    assert "providers.openai.base_url is not set" in capsys.readouterr().err


def test_the_installed_command_survives_a_broken_home_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".ember").mkdir(parents=True)
    (home / ".ember" / "config.yaml").write_text("default_model: [unclosed\n")
    result = subprocess.run(
        [sys.executable, "-m", "ember.cli", "configure", "show"],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        f"ember: error: {home}/.ember/config.yaml line 2: not valid YAML "
        "(expected ',' or ']', but got '<stream end>')"
    ]
