"""`ember models` and `ember check`: unknowns shown as unknown, typed names never echoed."""

import datetime
import decimal
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from ember import context
from ember.cli import main
from ember.cli.models import CHECK_PROMPT, catalog_table, check_report
from ember.errors import ErrorKind
from ember.llm import (
    Client,
    FinishReason,
    Message,
    ModelRef,
    ModelSpec,
    Money,
    Rates,
    Result,
    TextPart,
    Usage,
    catalog,
    using,
)
from ember.llm.fake import Fail, FakeProvider, Reply
from ember.llm.spec import LongContextRates

KEY = "sk-secret-abcdefgh1234"
RATES = Rates(
    input=decimal.Decimal("1.250"),
    cache_read=decimal.Decimal("0.125"),
    cache_write=None,
    output=decimal.Decimal("10"),
    source_url="https://example.com/pricing",
    as_of=datetime.date(2026, 11, 1),
)
PRICED = ModelSpec(
    ref=ModelRef("openai", "gpt-x"),
    wire_id="gpt-x",
    api="openai_responses",
    aliases=("gpt-x", "x"),
    context_window=400_000,
    max_output_tokens=128_000,
    rates=Rates(
        input=RATES.input,
        cache_read=RATES.cache_read,
        cache_write=None,
        output=RATES.output,
        source_url=RATES.source_url,
        as_of=RATES.as_of,
        long_context=LongContextRates(272_000, RATES),
    ),
)
INPUT_LIMITED = ModelSpec(
    ref=ModelRef("gemini", "g"), wire_id="g", api="gemini", max_input_tokens=1_048_576
)


@pytest.fixture(autouse=True)
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EMBER_CONFIG", str(path))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield path
    context.reload()


def fake_client(fake: FakeProvider) -> Client:
    return Client(provider_factory=fake.factory, sleep=lambda seconds: None)


def result(*, usage: Usage, cost: Money | None, request_id: str | None) -> Result:
    return Result(
        text="ok\nmore",
        reasoning="",
        tool_calls=(),
        finish_reason=FinishReason.STOP,
        usage=usage,
        cost=cost,
        model=PRICED.ref,
        request_id=request_id,
        latency_s=1.234,
        message=Message("assistant", (TextPart("ok\nmore"),)),
    )


def test_catalog_table_aligns_columns_and_marks_unknowns() -> None:
    assert catalog_table([PRICED, INPUT_LIMITED]).splitlines() == [
        "MODEL         ALIASES             LIMIT  MAX OUTPUT  INPUT $/M  OUTPUT $/M"
        "  LONG PROMPTS   PRICES AS OF",
        "openai/gpt-x  gpt-x, x    400,000 total     128,000       1.25       10.00"
        "  above 272,000  2026-11-01",
        "gemini/g      -         1,048,576 input           -          -           -"
        "  -              -",
    ]


def test_ember_models_lists_every_catalog_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    lines = capsys.readouterr().out.splitlines()
    specs = list(catalog.default())
    assert lines[0].startswith("MODEL")
    assert [line.split()[0] for line in lines[1:]] == [str(spec.ref) for spec in specs]
    unpriced = [spec for spec in specs if spec.rates is None]
    for spec in unpriced:
        row = next(line for line in lines if line.startswith(f"{spec.ref} "))
        assert "$" not in row and " 0.00 " not in row


def test_check_report_states_every_field() -> None:
    usage = Usage(input=10, cache_read=2, output=3, reasoning=1)
    report = check_report(result(usage=usage, cost=Money(450), request_id="req_1"), "openai/gpt-x")
    assert report.splitlines() == [
        "openai/gpt-x answered in 1.23 s",
        "  finish reason  stop",
        "  request id     req_1",
        "  tokens         12 in, 3 out (1 reasoning)",
        "  cost           $0.00000045",
        "  text           ok",
    ]


def test_check_report_never_shows_unknown_cost_or_usage_as_zero() -> None:
    report = check_report(result(usage=Usage(measured=False), cost=None, request_id=None), "m")
    assert "request id     not sent by the provider" in report
    assert "tokens         not reported by the provider" in report
    assert "cost           unknown: the model has no known price" in report
    assert "$0" not in report


def test_check_sends_one_short_request_and_reports_it(capsys: pytest.CaptureFixture[str]) -> None:
    usage = Usage(input=12, output=1)
    fake = FakeProvider(default=Reply("ok", usage=usage, request_id="req_9"))
    with fake_client(fake) as client, using(client):
        assert main(["check", "gpt-5.5"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r"openai/gpt-5\.5 answered in \d+\.\d\d s", out[0])
    spec = catalog.default().get("openai/gpt-5.5")
    assert spec is not None and spec.rates is not None
    assert out[1:] == [
        "  finish reason  stop",
        "  request id     req_9",
        "  tokens         12 in, 1 out (0 reasoning)",
        f"  cost           {spec.rates.cost(usage)}",
        "  text           ok",
    ]
    (seen,) = fake.requests
    assert seen.model == ModelRef("openai", "gpt-5.5")
    assert seen.request.messages == (Message.user(CHECK_PROMPT),)


def test_a_failed_check_is_one_line_from_the_error_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeProvider(default=Fail(ErrorKind.AUTH, status=401))
    with fake_client(fake) as client, using(client):
        assert main(["check", "openai/gpt-5.5"]) == 1
    assert capsys.readouterr().err == (
        "ember: error: openai/gpt-5.5: auth: scripted failure status=401\n"
    )


def test_a_model_outside_the_catalog_is_sent_but_not_repeated(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file.write_text(
        "providers:\n  local:\n    kind: openai_compatible\n    base_url: http://h/v1\n"
    )
    fake = FakeProvider(default=Fail(ErrorKind.NOT_FOUND, status=404))
    with fake_client(fake) as client, using(client):
        assert main(["check", f"local/{KEY}"]) == 1
    err = capsys.readouterr().err
    assert (
        err
        == "ember: error: local/… (not in the catalog): not_found: scripted failure status=404\n"
    )
    assert [seen.wire_id for seen in fake.requests] == [KEY]


@pytest.mark.parametrize("model", [KEY, f"nobody/{KEY}", "gpt-5.5x", "openai/", "/gpt-5.5"])
def test_an_unknown_model_is_refused_by_role_before_any_request(
    capsys: pytest.CaptureFixture[str], model: str
) -> None:
    fake = FakeProvider()
    with fake_client(fake) as client, using(client):
        assert main(["check", model]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ember: error: MODEL is not a catalog model or alias")
    assert model not in err
    assert fake.requests == ()
