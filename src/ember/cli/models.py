"""``ember models`` and ``ember check``: what ember can call, and whether a call works."""

import argparse
import decimal
import sys
from collections.abc import Iterable

import ember
from ember import context
from ember.errors import ConfigError, ModelError
from ember.llm import ModelRef, ModelSpec, Result, catalog

__all__ = ["CHECK_PROMPT", "catalog_table", "check", "check_report", "list_models"]

CHECK_PROMPT = "Reply with the single word: ok"
_UNKNOWN = "-"
# The model is named by its role: it came from the command line and may be a key.
_UNKNOWN_MODEL = (
    "MODEL is not a catalog model or alias, nor provider/name for a configured provider; "
    "ember models lists the catalog"
)


def list_models(args: argparse.Namespace) -> int:
    """Prints the model catalog."""
    del args
    print(catalog_table(catalog.default()))
    return 0


def check(args: argparse.Namespace) -> int:
    """Sends one short request and reports what came back, or why it failed."""
    ref, label = _check_target(args.model)
    try:
        result = ember.models(ref, CHECK_PROMPT).result()
    except ModelError as error:
        print(f"ember: error: {label}: {_failure(error)}", file=sys.stderr)
        return 1
    print(check_report(result, label))
    return 0


def _check_target(name: str) -> tuple[ModelRef, str]:
    """Resolves the model to check, and how output may name it.

    A catalog model is named by its catalog id. Any other ``provider/name``
    for a configured provider is sent as given but not repeated in output.
    """
    spec = catalog.default().get(name)
    if spec is not None:
        return spec.ref, str(spec.ref)
    provider, slash, model = name.partition("/")
    configured = context.current().providers
    if slash and model and model == model.strip() and provider in configured:
        return ModelRef(provider, model), f"{provider}/… (not in the catalog)"
    raise ConfigError(_UNKNOWN_MODEL)


def _failure(error: ModelError) -> str:
    """Describes a failed call from its fields; the model name is left out."""
    parts = [f"{error.kind.value}: {error.reason}"]
    if error.status is not None:
        parts.append(f"status={error.status}")
    if error.request_id is not None:
        parts.append(f"request_id={error.request_id}")
    if error.retry_after_s is not None:
        parts.append(f"retry_after_s={error.retry_after_s:g}")
    return " ".join(parts)


def catalog_table(specs: Iterable[ModelSpec]) -> str:
    """Formats catalog entries as aligned columns; unknown values show as ``-``.

    LIMIT is the combined prompt-and-output window (``total``) or the prompt
    limit (``input``), whichever the provider states. LONG PROMPTS names the
    prompt size above which the model's prices are higher than shown.
    """
    header = (
        "MODEL",
        "ALIASES",
        "LIMIT",
        "MAX OUTPUT",
        "INPUT $/M",
        "OUTPUT $/M",
        "LONG PROMPTS",
        "PRICES AS OF",
    )
    numeric = (False, False, True, True, True, True, False, False)
    rows = [header, *(_catalog_row(spec) for spec in specs)]
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return "\n".join(
        "  ".join(
            cell.rjust(width) if right else cell.ljust(width)
            for cell, width, right in zip(row, widths, numeric, strict=True)
        ).rstrip()
        for row in rows
    )


def _catalog_row(spec: ModelSpec) -> tuple[str, ...]:
    rates = spec.rates
    long_context = None if rates is None else rates.long_context
    return (
        str(spec.ref),
        ", ".join(spec.aliases) or _UNKNOWN,
        _limit(spec),
        _tokens(spec.max_output_tokens),
        _UNKNOWN if rates is None else _dollars(rates.input),
        _UNKNOWN if rates is None else _dollars(rates.output),
        _UNKNOWN if long_context is None else f"above {long_context.above_prompt_tokens:,}",
        _UNKNOWN if rates is None else rates.as_of.isoformat(),
    )


def _limit(spec: ModelSpec) -> str:
    if spec.context_window is not None:
        return f"{spec.context_window:,} total"
    if spec.max_input_tokens is not None:
        return f"{spec.max_input_tokens:,} input"
    return _UNKNOWN


def _tokens(count: int | None) -> str:
    return _UNKNOWN if count is None else f"{count:,}"


def _dollars(rate: decimal.Decimal) -> str:
    """Dollars with at least two decimals and no rounding: 0.10, 10.00, 0.125."""
    whole, _, fraction = format(rate.normalize(), "f").partition(".")
    return f"{whole}.{fraction.ljust(2, '0')}"


def check_report(result: Result, label: str) -> str:
    """Describes a successful check call: what came back and what it cost.

    Args:
        result: The call's result.
        label: How to name the model; see `check`.
    """
    usage = result.usage
    tokens = (
        f"{usage.prompt_tokens:,} in, {usage.output:,} out ({usage.reasoning:,} reasoning)"
        if usage.measured
        else "not reported by the provider"
    )
    cost = "unknown: the model has no known price" if result.cost is None else str(result.cost)
    lines = [
        f"{label} answered in {result.latency_s:.2f} s",
        f"  finish reason  {result.finish_reason.value}",
        f"  request id     {result.request_id or 'not sent by the provider'}",
        f"  tokens         {tokens}",
        f"  cost           {cost}",
        f"  text           {_first_line(result.text)}",
    ]
    return "\n".join(lines)


def _first_line(text: str, limit: int = 200) -> str:
    line = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return line if len(line) <= limit else line[: limit - 1] + "…"
