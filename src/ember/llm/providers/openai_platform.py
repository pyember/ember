"""What OpenAI's two wire protocols share: values, usage and error codes.

Chat Completions and Responses take the same reasoning efforts, report usage
with the same nesting under different field names, and fail with the same
error object. This module holds those once; each codec keeps its own shapes.

Wire reference, read 2026-11-07:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://developers.openai.com/api/reference/resources/responses/methods/create
https://developers.openai.com/api/docs/guides/error-codes
"""

import base64
import dataclasses
import json
from collections.abc import Mapping

from ember.errors import ErrorKind
from ember.llm.providers.wire import (
    Failure,
    count,
    identifier,
    json_object,
    kind_for_status,
    malformed,
)
from ember.llm.request import ImagePart, JsonValue
from ember.llm.result import Usage

__all__ = [
    "EFFORTS",
    "SCHEMA_NAME",
    "UsageFields",
    "arguments_json",
    "classify",
    "error_failure",
    "image_url",
    "usage",
]

#: Values ``reasoning_effort`` (Chat) and ``reasoning.effort`` (Responses) accept.
EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})

#: The name ember gives the caller's output schema in both protocols.
SCHEMA_NAME = "output"

# Billing, spend and quota failures: retrying cannot restore access. The
# error-codes guide lists these codes and says their type can be
# ``insufficient_quota``.
_QUOTA_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted"})
_QUOTA_SUFFIXES = ("_spend_limit_exceeded", "_usage_limit_exceeded")
_CODE_KINDS: Mapping[str, ErrorKind] = {
    "rate_limit_exceeded": ErrorKind.RATE_LIMITED,
    "server_error": ErrorKind.SERVER,
    "vector_store_timeout": ErrorKind.SERVER,
}


def image_url(part: ImagePart) -> str:
    """Returns the image's URL, or its bytes as a ``data:`` URL."""
    if part.url is not None:
        return part.url
    assert part.data is not None  # ImagePart holds exactly one of url and data.
    return f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}"


def arguments_json(arguments: Mapping[str, JsonValue]) -> str:
    """Serializes tool-call arguments the way both protocols carry them."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def code_kind(code: object, error_type: object) -> ErrorKind | None:
    """Maps an OpenAI error ``code`` or ``type`` to a kind, when it names one."""
    for value in (code, error_type):
        if not isinstance(value, str):
            continue
        if value in _QUOTA_CODES or value.endswith(_QUOTA_SUFFIXES):
            return ErrorKind.QUOTA
        if value in _CODE_KINDS:
            return _CODE_KINDS[value]
    return None


def error_failure(
    error: Mapping[str, object],
    status: int | None,
    unexplained: ErrorKind = ErrorKind.SERVER,
) -> Failure:
    """Classifies an OpenAI error object, from an HTTP body or inside a stream.

    Args:
        error: The ``error`` object.
        status: The HTTP status, or ``None`` inside a stream.
        unexplained: The kind when neither the code nor a status decides.

    Returns:
        The failure: the code's kind when it names one, else the status's,
        else ``unexplained``.
    """
    code, error_type = error.get("code"), error.get("type")
    kind = code_kind(code, error_type)
    if kind is None:
        kind = unexplained if status is None else kind_for_status(status)
    return Failure(kind, identifier(code) or identifier(error_type))


def classify(status: int, body: object) -> Failure:
    """Classifies an HTTP error response from either OpenAI protocol."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return Failure(kind_for_status(status))
    return error_failure(error, status)


@dataclasses.dataclass(frozen=True, slots=True)
class UsageFields:
    """Where one protocol keeps its totals; details sit at ``<name>_details``.

    Attributes:
        prompt: The prompt total, for example ``prompt_tokens``.
        output: The output total, for example ``completion_tokens``.
    """

    prompt: str
    output: str


def usage(value: object, fields: UsageFields) -> Usage:
    """Maps an OpenAI usage object; no object means nothing was measured.

    The prompt total includes cached and cache-written tokens, and the output
    total includes reasoning tokens. Absent or null detail objects and counts
    are zero, as the reference documents their defaults.
    """
    if value is None:
        return Usage(measured=False)
    totals = json_object(value, "usage")
    prompt = count(totals.get(fields.prompt), f"usage.{fields.prompt}")
    output = count(totals.get(fields.output), f"usage.{fields.output}")
    prompt_details = _details(totals, f"{fields.prompt}_details")
    output_details = _details(totals, f"{fields.output}_details")
    cache_read = _detail(prompt_details, "cached_tokens")
    cache_write = _detail(prompt_details, "cache_write_tokens")
    reasoning = _detail(output_details, "reasoning_tokens")
    if cache_read + cache_write > prompt or reasoning > output:
        raise malformed("usage details exceed their totals")
    return Usage(
        input=prompt - cache_read - cache_write,
        cache_read=cache_read,
        cache_write=cache_write,
        output=output,
        reasoning=reasoning,
    )


def _details(totals: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = totals.get(key)
    return {} if value is None else json_object(value, f"usage.{key}")


def _detail(details: Mapping[str, object], key: str) -> int:
    value = details.get(key)
    return 0 if value is None else count(value, f"usage.{key}")
