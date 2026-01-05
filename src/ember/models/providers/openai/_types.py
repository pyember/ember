from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

MessageDict = Mapping[str, object]
MessageList = Sequence[MessageDict]


class ResponsesDelta(TypedDict, total=False):
    text: str


class ResponsesEvent(TypedDict, total=False):
    type: str
    delta: ResponsesDelta | str | None
    error: object | None
    usage_delta: object | None
