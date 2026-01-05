"""Unit tests for `has_jax_arrays` detection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ember.xcs.compiler.analysis import has_jax_arrays


@dataclass
class Payload:
    values: list[dict[str, object]]


def test_has_jax_arrays_detects_nested_dict() -> None:
    tensor = jnp.arange(3.0)
    assert has_jax_arrays((), {"payload": {"tensor": tensor}})


def test_has_jax_arrays_detects_dataclass_payload() -> None:
    container = Payload(values=[{"tensor": jnp.ones((2,))}])
    assert has_jax_arrays((container,), {})


def test_has_jax_arrays_ignores_plain_scalars() -> None:
    assert not has_jax_arrays(({"text": "hello"},), None)


def test_has_jax_arrays_detects_shape_dtype_struct() -> None:
    struct = jax.ShapeDtypeStruct((4,), jnp.float32)
    assert has_jax_arrays((struct,), None)
