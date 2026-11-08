"""The Server-Sent Events parser every streaming provider shares.

Parses a byte stream into events following the WHATWG HTML "event stream
interpretation" rules: lines end in CRLF, LF or CR (a CRLF may be split across
chunks); a leading byte order mark is dropped; a line starting with ``:`` is a
comment; ``field: value`` loses one leading space from the value; repeated
``data`` lines join with ``\\n``; a blank line dispatches the event; an event
with no data is not dispatched.

One deliberate difference from a browser ``EventSource``: an event still
pending when the stream ends is dispatched, not discarded. A browser drops it
because it will reconnect; ember never reconnects, and every codec checks for
its provider's terminal event and rejects malformed JSON, so a cut stream
still fails loudly while a server that omits the last blank line still works.

Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
"""

import codecs
import dataclasses
import re
from collections.abc import Iterable, Iterator

__all__ = ["SseEvent", "SseParser", "parse"]

_LINE_END = re.compile(r"\r\n|\r|\n")
_BOM = "﻿"


@dataclasses.dataclass(frozen=True, slots=True)
class SseEvent:
    """One dispatched event.

    Attributes:
        event: The ``event`` field, or ``"message"`` when the event had none.
        data: The ``data`` lines joined with ``\\n``.
        id: The last ``id`` field seen so far in the stream, if any.
    """

    event: str
    data: str
    id: str | None = None


class SseParser:
    """Incremental parser for one stream; feed it bytes as they arrive."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._partial_line = ""
        self._at_start = True
        self._drop_leading_lf = False
        self._event = ""
        self._data: list[str] = []
        self._last_id: str | None = None

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Consumes ``chunk`` and returns the events it completed.

        Raises:
            UnicodeDecodeError: The stream is not valid UTF-8.
        """
        return self._consume(self._decoder.decode(chunk))

    def close(self) -> list[SseEvent]:
        """Ends the stream and returns the events still pending.

        Raises:
            UnicodeDecodeError: The stream ended inside a UTF-8 sequence.
        """
        events = self._consume(self._decoder.decode(b"", final=True))
        if self._partial_line:
            self._process_line(self._partial_line)
            self._partial_line = ""
        pending = self._dispatch()
        if pending is not None:
            events.append(pending)
        return events

    def _consume(self, text: str) -> list[SseEvent]:
        if not text:
            return []
        if self._at_start:
            text = text.removeprefix(_BOM)
            self._at_start = False
        if self._drop_leading_lf:
            text = text.removeprefix("\n")
            self._drop_leading_lf = False
        buffer = self._partial_line + text
        events: list[SseEvent] = []
        start = 0
        for match in _LINE_END.finditer(buffer):
            line = buffer[start : match.start()]
            start = match.end()
            # A CR that ends the buffer may be the first half of a CRLF.
            self._drop_leading_lf = match.group() == "\r" and start == len(buffer)
            event = self._process_line(line)
            if event is not None:
                events.append(event)
        self._partial_line = buffer[start:]
        return events

    def _process_line(self, line: str) -> SseEvent | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        name, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\0" not in value:
            self._last_id = value
        return None

    def _dispatch(self) -> SseEvent | None:
        data, event = self._data, self._event
        self._data, self._event = [], ""
        if not data:
            return None
        return SseEvent(event=event or "message", data="\n".join(data), id=self._last_id)


def parse(chunks: Iterable[bytes]) -> Iterator[SseEvent]:
    """Yields the events in a stream of byte chunks.

    Raises:
        UnicodeDecodeError: The stream is not valid UTF-8.
    """
    parser = SseParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.close()
