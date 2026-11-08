"""Encoding items and outputs as JSON for the run records."""

import math

from ember.data import Example
from ember.llm.request import JsonValue

__all__ = ["to_json"]


def to_json(value: object) -> JsonValue:
    """Encodes an item or output for the records.

    Accepts `ember.data.Example`, and JSON values: strings, numbers, booleans,
    ``None``, and lists, tuples and string-keyed dicts of them.

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
    return _json_value(value)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot record {value}: JSON has no such number")
        return value
    if isinstance(value, list | tuple):
        return [_json_value(entry) for entry in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("cannot record a dict with non-string keys")
        return {key: _json_value(entry) for key, entry in value.items()}
    raise TypeError(
        f"cannot record a {type(value).__name__}; "
        "pass encode_item= or encode_output= to ember.eval.run"
    )
