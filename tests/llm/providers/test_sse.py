"""The SSE parser follows the WHATWG event-stream rules, plus one documented difference."""

import pytest

from ember.llm.providers.sse import SseEvent, SseParser, parse


def _events(*chunks: bytes) -> list[SseEvent]:
    return list(parse(chunks))


def test_multi_line_data_joins_with_newlines() -> None:
    assert _events(b"data: first\ndata: second\ndata:third\n\n") == [
        SseEvent("message", "first\nsecond\nthird")
    ]


def test_comments_are_ignored_and_event_names_apply_to_one_event() -> None:
    stream = b": keep-alive\nevent: ping\ndata: {}\n\n: another comment\ndata: x\n\n"
    assert _events(stream) == [SseEvent("ping", "{}"), SseEvent("message", "x")]


@pytest.mark.parametrize("ending", [b"\r\n", b"\r", b"\n"])
def test_every_line_ending_ends_lines(ending: bytes) -> None:
    stream = ending.join([b"event: a", b"data: 1", b"", b"data: 2", b"", b""])
    assert _events(stream) == [SseEvent("a", "1"), SseEvent("message", "2")]


def test_crlf_split_across_chunks_is_one_line_ending() -> None:
    # A blank line would dispatch early if "\r" and "\n" counted as two endings.
    assert _events(b"data: a\r", b"\ndata: b\r", b"\n\r\n") == [SseEvent("message", "a\nb")]


def test_final_event_without_trailing_blank_line_is_dispatched() -> None:
    assert _events(b"data: one\n\ndata: last") == [
        SseEvent("message", "one"),
        SseEvent("message", "last"),
    ]
    assert _events(b"data: last\n") == [SseEvent("message", "last")]


def test_only_one_leading_space_is_stripped_and_bare_fields_count() -> None:
    assert _events(b"data:  two spaces\n\n") == [SseEvent("message", " two spaces")]
    assert _events(b"data\n\n") == [SseEvent("message", "")]


def test_events_without_data_are_not_dispatched() -> None:
    assert _events(b"event: lonely\n\n\n\nid: 7\n\n") == []


def test_last_event_id_carries_forward_and_rejects_nul() -> None:
    events = _events(b"id: 1\ndata: a\n\ndata: b\n\nid: x\x00y\ndata: c\n\n")
    assert [e.id for e in events] == ["1", "1", "1"]


def test_leading_byte_order_mark_is_dropped() -> None:
    assert _events(b"\xef\xbb\xbfdata: x\n\n") == [SseEvent("message", "x")]


def test_multibyte_characters_split_across_chunks_decode() -> None:
    encoded = "data: café ✓\n\n".encode()
    split = encoded.index("é".encode()) + 1
    assert _events(encoded[:split], encoded[split:]) == [SseEvent("message", "café ✓")]


def test_invalid_utf8_raises() -> None:
    with pytest.raises(UnicodeDecodeError):
        _events(b"data: \xff\n\n")


def test_parser_returns_events_as_soon_as_their_blank_line_arrives() -> None:
    parser = SseParser()
    assert parser.feed(b"data: a\n") == []
    assert parser.feed(b"\ndata: b") == [SseEvent("message", "a")]
    assert parser.close() == [SseEvent("message", "b")]


def test_documented_stream_with_comment_parses(kit) -> None:
    events = kit.events("openai/responses_stream_text.sse")
    assert [e.event for e in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
