"""The catalog: one table, read from official pages, resolved without guessing."""

import ast
import datetime
import decimal
import importlib.resources
import urllib.parse
from pathlib import Path

import pytest

from ember.context import ProviderSettings
from ember.errors import ErrorKind, ModelError
from ember.llm import catalog
from ember.llm.catalog import Catalog, CatalogError
from ember.llm.ref import ModelRef
from ember.llm.result import Money, Usage

SRC = Path(__file__).resolve().parents[2] / "src" / "ember"
OFFICIAL_HOSTS = {"platform.claude.com", "developers.openai.com", "aistudio.google.com"}
PROVIDERS = {
    "openai": ProviderSettings("openai", "openai"),
    "anthropic": ProviderSettings("anthropic", "anthropic"),
    "local": ProviderSettings("local", "openai_compatible", base_url="http://localhost:11434/v1"),
}


def test_packaged_catalog_ships_as_package_data() -> None:
    resource = importlib.resources.files("ember.llm").joinpath("catalog.toml")
    assert resource.is_file()
    assert len(catalog.default()) == len(Catalog.from_toml(resource.read_text()))


def test_every_value_cites_an_official_page_and_a_date() -> None:
    for spec in catalog.default():
        assert spec.known
        urls = [*spec.sources, *([spec.rates.source_url] if spec.rates else [])]
        hosts = {urllib.parse.urlsplit(url).hostname for url in urls}
        assert hosts <= OFFICIAL_HOSTS, f"{spec.ref} cites {hosts - OFFICIAL_HOSTS}"
        assert spec.as_of == datetime.date(2026, 11, 7)
        assert spec.rates is None or spec.rates.as_of == datetime.date(2026, 11, 7)


def test_ids_aliases_and_snapshot_ids_resolve_to_one_spec() -> None:
    table = catalog.default()
    spec = table.resolve("anthropic/claude-opus-4-5", PROVIDERS)
    assert spec.wire_id == "claude-opus-4-5-20251101"
    assert table.resolve("claude-opus-4-5", PROVIDERS) is spec
    assert table.resolve("anthropic/claude-opus-4-5-20251101", PROVIDERS) is spec
    assert table.resolve(ModelRef("anthropic", "claude-opus-4-5"), PROVIDERS) is spec


def test_alias_names_the_model_the_provider_page_names() -> None:
    # OpenAI's GPT-5.6 Sol page: "The `gpt-5.6` alias routes requests to GPT-5.6 Sol."
    spec = catalog.default().resolve("gpt-5.6", PROVIDERS)
    assert spec.ref == ModelRef("openai", "gpt-5.6-sol")


@pytest.mark.parametrize(
    ("model", "api"),
    [
        ("openai/gpt-5", "openai_responses"),
        ("anthropic/claude-3-haiku", "anthropic_messages"),
        ("local/llama3.2:1b", "openai_chat"),
        ("local/meta-llama/Llama-3.1-8B", "openai_chat"),
    ],
)
def test_uncatalogued_model_on_a_configured_provider_is_sent_as_named(model: str, api: str) -> None:
    spec = catalog.default().resolve(model, PROVIDERS)
    provider, _, name = model.partition("/")
    assert spec.ref == ModelRef(provider, name)
    assert (spec.wire_id, spec.api, spec.known) == (name, api, False)
    assert (spec.rates, spec.context_window, spec.max_output_tokens) == (None, None, None)


@pytest.mark.parametrize("name", ["gpt-5.5-turbo", "gpt5.5", "opus", "llama3.2:1b"])
def test_bare_unknown_names_are_not_found_and_never_guessed(name: str) -> None:
    with pytest.raises(ModelError) as caught:
        catalog.default().resolve(name, PROVIDERS)
    error = caught.value
    assert (error.kind, error.provider, error.model) == (ErrorKind.NOT_FOUND, "", name)
    assert not error.retryable
    assert str(error).endswith(f"model={name}")
    assert ".py" not in str(error) and str(SRC) not in str(error)


def test_a_suffix_on_a_known_id_is_a_different_model() -> None:
    spec = catalog.default().resolve("openai/gpt-5.5-turbo", PROVIDERS)
    assert (spec.wire_id, spec.known) == ("gpt-5.5-turbo", False)


def test_unconfigured_provider_is_not_found() -> None:
    with pytest.raises(ModelError) as caught:
        catalog.default().resolve("mystery/model-1", PROVIDERS)
    assert caught.value.kind is ErrorKind.NOT_FOUND
    assert "configured providers: anthropic, local, openai" in str(caught.value)


def test_long_context_rates_start_above_the_published_threshold() -> None:
    # Anthropic's pricing page: Haiku 5.5 input is $0.10/MTok for prompts up to
    # 100,000 tokens and $0.50/MTok for prompts over 100,000 tokens.
    rates = catalog.default().resolve("claude-haiku-5-5", PROVIDERS).rates
    assert rates is not None
    assert rates.cost(Usage(input=100_000)) == Money(10_000_000)
    assert rates.cost(Usage(input=100_001)) == Money(50_000_500)


def test_loading_happens_once_until_reset() -> None:
    first = catalog.default()
    assert catalog.default() is first
    catalog.reset()
    assert catalog.default() is not first


_ENTRY = """
[[model]]
id = "openai/m1"
wire_id = "m1"
api = "openai_responses"
aliases = ["{alias}"]
accepts = ["max_output_tokens"]
reasoning_efforts = []
sources = ["https://developers.openai.com/api/docs/models/m1"]
as_of = 2026-11-07
"""


def test_two_models_claiming_one_name_fail_at_load() -> None:
    second = _ENTRY.replace('"openai/m1"', '"openai/m2"').replace('"m1"', '"m2"')
    with pytest.raises(CatalogError, match="'shared' names both openai/m1 and openai/m2"):
        Catalog.from_toml(_ENTRY.format(alias="shared") + second.format(alias="shared"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("as_of = 2026-11-07", "as_of = 2026-11-07\ncontext = 5"), "unknown keys"),
        (('aliases = ["m"]', 'aliases = ["openai/m"]'), "bare names"),
        (('accepts = ["max_output_tokens"]', 'accepts = ["temprature"]'), "unknown settings"),
        (('api = "openai_responses"', 'api = "responses"'), "api must be one of"),
    ],
)
def test_malformed_entries_fail_at_load(change: tuple[str, str], message: str) -> None:
    text = _ENTRY.format(alias="m").replace(*change)
    with pytest.raises(CatalogError, match=message):
        Catalog.from_toml(text)


def test_rates_must_be_exact_decimal_strings() -> None:
    priced = _ENTRY.format(alias="m") + (
        '[model.rates]\ninput = 1.1\noutput = "2"\n'
        'source_url = "https://developers.openai.com/api/docs/pricing"\nas_of = 2026-11-07\n'
    )
    with pytest.raises(CatalogError, match="decimal string"):
        Catalog.from_toml(priced)
    rates = Catalog.from_toml(priced.replace("input = 1.1", 'input = "1.1"')).get("m")
    assert rates is not None and rates.rates is not None
    assert rates.rates.input == decimal.Decimal("1.1")
    assert rates.rates.cache_read is None


def _string_constants_outside_docstrings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def test_no_code_outside_the_catalog_names_a_catalog_model() -> None:
    names = {str(spec.ref) for spec in catalog.default()}
    names |= {spec.wire_id for spec in catalog.default()}
    names |= {alias for spec in catalog.default() for alias in spec.aliases}
    offenders = {
        f"{path.relative_to(SRC)}: {value}"
        for path in SRC.rglob("*.py")
        for value in _string_constants_outside_docstrings(path)
        if value in names
    }
    assert not offenders, sorted(offenders)
