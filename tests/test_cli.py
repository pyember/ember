"""The ember command: parses first, never prints or accepts keys, repairs typos."""

import ast
import getpass
import re
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

import ember.cli
from ember import context
from ember.cli import main

KEY = "sk-secret-abcdefgh1234"
SPAWNS = re.compile(r"input|system|popen|fork|forkpty|exec[lv]p?e?|spawn[lv]p?e?|posix_spawnp?")


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
    err = capsys.readouterr().err
    assert "PROVIDER is not a configured provider; known: anthropic, gemini, openai." in err
    assert "opnai" not in err
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
        (["set", "local", "base_url", "http://h/v1"], "PROVIDER is not a configured provider"),
        (["set", "local", "kind", "bogus"], "KIND must be one of openai, anthropic, gemini"),
        (["set", "openai", "kind", "anthropic"], "config.yaml: provider 'openai' is openai"),
    ],
)
def test_changes_the_library_would_reject_are_not_written(
    config_file: Path, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    assert main(["configure", *argv]) == 1
    assert message in capsys.readouterr().err
    assert not config_file.exists()


def test_an_openai_compatible_provider_is_set_up_field_by_field(config_file: Path) -> None:
    assert main(["configure", "set", "local", "kind", "openai_compatible"]) == 0
    assert main(["configure", "set", "local", "base_url", "http://localhost:11434/v1"]) == 0
    context.reload()
    local = context.current().provider("local")
    assert (local.kind, local.base_url) == ("openai_compatible", "http://localhost:11434/v1")


def test_unset_refuses_to_break_a_config_that_loads(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "providers:\n  local:\n    kind: openai_compatible\n    base_url: http://h/v1\n"
    config_file.write_text(text)
    assert main(["configure", "unset", "local", "kind"]) == 1
    err = capsys.readouterr().err
    assert "nothing was saved" in err and "provider 'local' needs kind" in err
    assert config_file.read_text() == text
    assert main(["configure", "unset", "local"]) == 0
    assert "providers" not in config_file.read_text()


def test_unset_repairs_a_config_that_does_not_load_and_says_what_remains(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text("providers:\n  local:\n    base_url: http://h/v1\n    api_kye: y\n")
    assert main(["configure", "unset", "local", "api_kye"]) == 1
    out, err = capsys.readouterr()
    assert out.startswith("removed providers.local.api_kye")
    assert f"the config still does not load: {config_file}: provider 'local' needs kind" in err
    assert "api_kye" not in config_file.read_text()


@pytest.mark.usefixtures("no_prompt")
@pytest.mark.parametrize(
    "argv",
    [
        ["configure", "set-key", "openai", KEY],
        ["configure", "set-key", "openai", f"--key={KEY}"],
        ["configure", "unset", "openai", "api_key", KEY],
    ],
)
def test_extra_arguments_are_refused_without_echoing_them(
    config_file: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2
    out, err = capsys.readouterr()
    assert KEY not in out and KEY not in err
    assert "unexpected extra arguments (not shown)" in err
    assert not config_file.exists()


def test_set_key_without_a_terminal_saves_nothing(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_terminal(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(getpass, "getpass", no_terminal)
    assert main(["configure", "set-key", "openai"]) == 1
    assert capsys.readouterr().err == "ember: error: no key entered; nothing was saved\n"
    assert not config_file.exists()


def test_context_errors_name_the_file(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text("providers:\n  local:\n    kind: bogus\n")
    assert main(["configure", "show"]) == 1
    err = capsys.readouterr().err
    assert err == f"ember: error: {config_file}: provider 'local' has unknown kind 'bogus'; " + (
        "use openai, anthropic, gemini, openai_compatible\n"
    )


def test_a_file_with_comments_is_not_rewritten(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "providers:\n  openai:  # work account\n    base_url: http://h/v1\n"
    config_file.write_text(text)
    assert main(["configure", "set", "openai", "api_key_env", "WORK_KEY"]) == 1
    assert f"{config_file} line 2: the file has comments" in capsys.readouterr().err
    assert config_file.read_text() == text


def test_configure_help_says_comments_block_rewrites(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["configure", "--help"])
    assert "comments" in capsys.readouterr().out


def test_unset_of_something_not_set_is_an_error(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["configure", "unset", "openai", "base_url"]) == 1
    assert f"PROVIDER is not in {config_file}; providers there: none" in capsys.readouterr().err
    config_file.write_text("providers:\n  openai:\n    base_url: http://h/v1\n")
    assert main(["configure", "unset", "openai", "api_key"]) == 1
    assert "FIELD is not set for openai; fields set there: base_url" in capsys.readouterr().err


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
    assert len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith(
        f"ember: error: {home}/.ember/config.yaml line 2: not valid YAML"
    )


def test_cli_data_and_eval_never_start_processes_or_prompt() -> None:
    # v0.1 ran `npx -y @ember-ai/setup`, or blocked on input(), when a key was missing.
    source = Path(ember.cli.__file__).parents[1]
    paths = [path for unit in ("cli", "data", "eval") for path in (source / unit).rglob("*.py")]
    offences = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            if {"subprocess", "pty", "webbrowser"} & {name.split(".")[0] for name in modules}:
                offences.append(f"{path.name}:{node.lineno} imports {modules}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                if SPAWNS.fullmatch(name):
                    offences.append(f"{path.name}:{node.lineno} calls {name}")
    assert paths and not offences, offences


# Every subcommand with every argument filled in; the key is put in each slot in turn.
COMMANDS = (
    ("configure", "show"),
    ("configure", "set-key", "local"),
    ("configure", "set", "local", "base_url", "http://h/v1"),
    ("configure", "set", "local", "kind", "openai_compatible"),
    ("configure", "set", "local", "api_key_env", "LOCAL_API_KEY"),
    ("configure", "unset", "local", "base_url"),
    ("configure", "unset", "local"),
)


# Every option the parsers define, in full and abbreviated, plus one they do not.
FLAGS = ("--version", "--vers", "--help", "--he", "-h", "--x")


def key_in_every_slot() -> list[tuple[str, ...]]:
    cases: set[tuple[str, ...]] = set()
    for command in COMMANDS:
        for slot in range(len(command)):
            cases.add((*command[:slot], KEY, *command[slot + 1 :]))
        for slot in range(len(command) + 1):
            for token in (KEY, f"--{KEY}", f"-{KEY}", *(f"{flag}={KEY}" for flag in FLAGS)):
                cases.add((*command[:slot], token, *command[slot:]))
    return sorted(cases)


@pytest.mark.parametrize("argv", key_in_every_slot())
def test_output_never_echoes_a_key_given_in_any_argument_slot(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
) -> None:
    config_file.write_text(
        "providers:\n  local:\n    kind: openai_compatible\n    base_url: http://h/v1\n"
    )
    type_key(monkeypatch, "sk-typed-at-the-prompt-0000")
    crash = ""
    try:
        main(list(argv))
    except SystemExit:
        pass
    except Exception as exc:  # Recorded, then asserted absent below.
        crash = "".join(traceback.format_exception(exc))
    out, err = capsys.readouterr()
    assert KEY not in out + err + crash
    assert crash == ""
    assert main(["configure", "show"]) == 0
    assert KEY not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_url", KEY, "providers.local.base_url must be an http or https URL"),
        ("base_url", "localhost:11434/v1", "providers.local.base_url must be an http or https URL"),
        ("api_key_env", KEY, "providers.local.api_key_env must be an environment variable name"),
    ],
)
def test_non_secret_fields_refuse_values_that_could_be_a_misplaced_key(
    config_file: Path, capsys: pytest.CaptureFixture[str], field: str, value: str, message: str
) -> None:
    text = "providers:\n  local:\n    kind: openai_compatible\n"
    config_file.write_text(text)
    assert main(["configure", "set", "local", field, value]) == 1
    err = capsys.readouterr().err
    assert message in err and value not in err
    assert config_file.read_text() == text


def test_a_provider_name_that_could_be_a_key_is_refused(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["configure", "set", KEY, "kind", "openai_compatible"]) == 1
    err = capsys.readouterr().err
    assert "provider names must be" in err and KEY not in err
    assert not config_file.exists()
