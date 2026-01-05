"""Reliability regressions for XCS: config, vmap outputs, grad tree_map.

These tests are surgical and focus on correctness without changing public APIs.
"""

import jax.numpy as jnp

from ember.xcs import grad, jit, vmap
from ember.xcs.config import Config


def test_jit_accepts_config_keyword():
    """jit(config=...) should be accepted and produce correct results."""

    def f(x):
        return x * 2.0

    jitted = jit(f, config=Config(profile=False))
    x = jnp.array(3.0)
    assert jitted(x) == 6.0


def test_vmap_mixed_outputs_stack_arrays_only():
    """vmap over mixed outputs stacks arrays and keeps strings as list."""

    def f(x, i):
        # Return array and heterogenous string output
        return x + 1.0, f"s{i}"

    xs = jnp.ones((4,))
    idx = jnp.arange(4)

    vmf = vmap(f)
    arr_out, str_out = vmf(xs, idx)

    assert arr_out.shape == (4,)
    assert isinstance(str_out, (list, tuple))
    assert len(str_out) == 4 and str_out[0].startswith("s")


def test_grad_tree_map_smoke():
    """grad wrapper uses jax.tree_util.tree_map and works for pure JAX."""

    def loss_fn(w):
        return jnp.sum(w**2)

    g = grad(loss_fn)
    w = jnp.array([1.0, -2.0, 3.0])
    dw = g(w)
    assert jnp.allclose(dw, 2.0 * w)
