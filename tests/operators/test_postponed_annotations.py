"""Operators defined under PEP 563 with `TYPE_CHECKING`-only types.

v0.1 resolved string annotations against the wrong module, so `@op` and
`Operator` raised on the first call here (and a downstream copy silently
skipped validation). v0.2 never reads annotations, so they cannot matter.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pydantic
import pytest

from ember import op
from ember.operators import Operator

if TYPE_CHECKING:
    from typing_extensions_that_do_not_exist import Doc  # type: ignore[import-not-found]


class Order(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    sku: str


@op
def summarize(input: Doc) -> Doc:
    return input


@dataclasses.dataclass(frozen=True, slots=True)
class Pricer(Operator["Order", float]):
    input_model = Order

    def forward(self, input: Order) -> float:
        return float(len(input.sku))


def test_op_runs_with_annotations_that_cannot_be_resolved() -> None:
    assert summarize("text") == "text"


def test_declared_validation_still_applies() -> None:
    assert Pricer()({"sku": "abc"}) == 3.0  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        Pricer()({"oops": 1})  # type: ignore[arg-type]
