import pytest

from ember.errors import ErrorKind, ModelError, NonSyntaxError


@pytest.mark.parametrize(
    ("kind", "retryable"),
    [
        (ErrorKind.RATE_LIMITED, True),
        (ErrorKind.OVERLOADED, True),
        (ErrorKind.SERVER, True),
        (ErrorKind.AUTH, False),
        (ErrorKind.BAD_REQUEST, False),
        (ErrorKind.NOT_FOUND, False),
        (ErrorKind.UNPRICED, False),
        (ErrorKind.QUOTA, False),
        (ErrorKind.MALFORMED, False),
    ],
)
def test_retryable_follows_kind(kind: ErrorKind, retryable: bool) -> None:
    error = ModelError(kind, provider="openai", model="gpt-5", reason="x")
    assert error.retryable is retryable


@pytest.mark.parametrize(
    "kind",
    [ErrorKind.TRANSPORT, ErrorKind.OVERLOADED, ErrorKind.RATE_LIMITED, ErrorKind.SERVER],
)
def test_no_error_is_retryable_after_the_caller_received_output(kind: ErrorKind) -> None:
    before = ModelError(kind, provider="p", model="m", reason="x")
    after = ModelError(kind, provider="p", model="m", reason="x", before_first_byte=False)
    assert before.retryable
    assert not after.retryable


def test_message_is_built_from_fields_only() -> None:
    error = ModelError(
        ErrorKind.RATE_LIMITED,
        provider="anthropic",
        model="claude-sonnet-4-5",
        reason="rate limited",
        status=429,
        request_id="req_1",
        retry_after_s=2.0,
    )
    assert str(error) == (
        "rate_limited: rate limited model=anthropic/claude-sonnet-4-5 status=429 "
        "request_id=req_1 retry_after_s=2"
    )


def test_non_syntax_error_points_at_the_column() -> None:
    error = NonSyntaxError("3E@", 3, "expected a model")
    assert isinstance(error, ValueError)
    assert str(error).splitlines()[-1] == "     ^"
