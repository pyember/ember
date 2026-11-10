"""Encoding items and outputs as JSON for the run records."""

import math
from collections.abc import Mapping

from ember.data import Example
from ember.llm.request import JsonValue
from ember.non import Answer

__all__ = ["json_value", "to_json"]


def to_json(value: object) -> JsonValue:
    """Encodes an item or output for the records.

    Accepts `ember.data.Example`; a NON graph's `ember.non.Answer`, recorded
    as ``{"text": ...}`` because its stage records hold results and errors,
    and every call behind them is recorded as an attempt anyway; and JSON
    values: strings, numbers, booleans, ``None``, and lists, tuples and
    string-keyed mappings of them.

    Raises:
        TypeError: ``value`` is of another type; pass ``encode_item`` or
            ``encode_output`` to `run` for it.
        ValueError: ``value`` holds a float that JSON cannot represent.
    """
    if isinstance(value, Example):
        return {
            "question": value.question,
            "answer": value.answer,
            "choices": list(value.choices),
            "metadata": dict(value.metadata),
        }
    if isinstance(value, Answer):
        return {"text": value.text}
    return json_value(value)


def json_value(value: object) -> JsonValue:
    """Copies a JSON value into plain JSON types: lists, dicts and scalars.

    Tuples become lists and any string-keyed mapping becomes a dict, so a
    value reads back from a run file exactly as it was recorded.

    Raises:
        TypeError: ``value`` holds something other than a JSON value.
        ValueError: ``value`` holds a float that JSON cannot represent.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot record {value}: JSON has no such number")
        return value
    if isinstance(value, list | tuple):
        return [json_value(entry) for entry in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("cannot record a mapping with non-string keys")
        return {key: json_value(entry) for key, entry in value.items()}
    raise TypeError(
        f"cannot record a {type(value).__name__}; "
        "pass encode_item= or encode_output= to ember.eval.run"
    )
