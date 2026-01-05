"""Tests for XCS operation classification and heuristic detection.

These tests verify:
1. Explicit markers take priority over heuristics
2. Ambiguous keywords are handled correctly
3. Module-aware matching works
4. Runtime type checking is correct
"""

from __future__ import annotations

import pytest

from ember.api.decorators import mark_hybrid, mark_orchestration, mark_tensor
from ember.xcs.compiler.analysis import (
    OperationSummary,
    analyze_operations,
    has_jax_arrays,
    is_jax_array,
)


class TestExplicitMarkers:
    """Test that explicit markers override heuristic detection."""

    def test_mark_tensor_overrides_heuristics(self):
        """@mark_tensor should force tensor-only classification."""

        @mark_tensor
        def my_model_function():
            # "model" would normally trigger orchestration detection
            return "result"

        summary = analyze_operations(my_model_function)
        assert summary.only_tensor_ops
        assert not summary.has_orchestration_ops
        assert "marked" in summary.tensor_ops

    def test_mark_orchestration_overrides_heuristics(self):
        """@mark_orchestration should force orchestration classification."""

        @mark_orchestration
        def compute_array():
            # "array" would normally trigger tensor detection
            return []

        summary = analyze_operations(compute_array)
        assert summary.only_orchestration_ops
        assert not summary.has_tensor_ops
        assert "marked" in summary.orchestration_ops

    def test_mark_hybrid_sets_both(self):
        """@mark_hybrid should set both tensor and orchestration."""

        @mark_hybrid
        def mixed_function():
            return "result"

        summary = analyze_operations(mixed_function)
        assert summary.is_hybrid
        assert summary.has_tensor_ops
        assert summary.has_orchestration_ops


class TestAmbiguousKeywords:
    """Test that ambiguous keywords don't cause false positives."""

    def test_model_keyword_not_misclassified(self):
        """'model' alone should not trigger orchestration detection."""

        def use_torch_model(model, x):
            # This is a PyTorch model, not an LLM
            return model(x)

        summary = analyze_operations(use_torch_model)
        # "model" was removed from orchestration keywords
        assert not summary.has_orchestration_ops or summary.orchestration_ops == set()

    def test_call_keyword_not_misclassified(self):
        """'call' alone should not trigger orchestration detection."""

        def call_function(func, args):
            return func(*args)

        summary = analyze_operations(call_function)
        assert not summary.has_orchestration_ops or summary.orchestration_ops == set()

    def test_api_keyword_not_misclassified(self):
        """'api' alone should not trigger orchestration detection."""

        def make_api_request(api_client, endpoint):
            return api_client.get(endpoint)

        summary = analyze_operations(make_api_request)
        # "api" was removed - too ambiguous
        assert not summary.has_orchestration_ops or summary.orchestration_ops == set()


class TestModuleAwareMatching:
    """Test that module prefixes are used for disambiguation."""

    def test_torch_module_detected_as_tensor(self):
        """Functions with torch.* calls should be classified as tensor."""

        def use_torch():
            import torch

            return torch.tensor([1, 2, 3])

        summary = analyze_operations(use_torch)
        assert summary.has_tensor_ops

    def test_jax_module_detected_as_tensor(self):
        """Functions with jax.* calls should be classified as tensor."""

        def use_jax():
            import jax.numpy as jnp

            return jnp.array([1, 2, 3])

        summary = analyze_operations(use_jax)
        assert summary.has_tensor_ops

    def test_openai_detected_as_orchestration(self):
        """Functions with openai.* calls should be classified as orchestration."""

        def use_openai():
            import openai

            return openai.ChatCompletion.create()

        summary = analyze_operations(use_openai)
        assert summary.has_orchestration_ops

    def test_anthropic_detected_as_orchestration(self):
        """Functions with anthropic.* calls should be classified as orchestration."""

        def use_anthropic():
            import anthropic

            return anthropic.Anthropic().messages.create()

        summary = analyze_operations(use_anthropic)
        assert summary.has_orchestration_ops


class TestUnambiguousKeywords:
    """Test that clear keywords still work correctly."""

    def test_llm_keyword_detected(self):
        """'llm' should trigger orchestration detection."""

        def call_llm(prompt):
            return llm_client.complete(prompt)

        summary = analyze_operations(call_llm)
        assert summary.has_orchestration_ops

    def test_completion_keyword_detected(self):
        """'completion' should trigger orchestration detection."""

        def get_completion(text):
            return create_completion(text)

        summary = analyze_operations(get_completion)
        assert summary.has_orchestration_ops

    def test_numpy_keyword_detected(self):
        """'numpy' should trigger tensor detection."""

        def use_numpy():
            import numpy as np

            return np.array([1, 2, 3])

        summary = analyze_operations(use_numpy)
        assert summary.has_tensor_ops

    def test_matmul_keyword_detected(self):
        """'matmul' should trigger tensor detection."""

        def matrix_multiply(a, b):
            return matmul(a, b)

        summary = analyze_operations(matrix_multiply)
        assert summary.has_tensor_ops


class TestFallbackBehavior:
    """Test behavior when analysis cannot determine type."""

    def test_unknown_function_returns_unknown_not_hybrid(self):
        """Functions with no source should return UNKNOWN, not hybrid.

        This is a key semantic distinction: unknown means we couldn't
        analyze the function, not that it contains both tensor and
        orchestration operations.
        """
        from ember.xcs.compiler.analysis import OpKind, analyze_operations_v2

        # Built-in functions don't have accessible source
        decision = analyze_operations_v2(len)

        # Should be UNKNOWN, not HYBRID
        assert decision.kind == OpKind.UNKNOWN
        # Legacy summary should have empty sets (not sentinel "unknown" strings)
        summary = analyze_operations(len)
        assert not summary.has_tensor_ops
        assert not summary.has_orchestration_ops

    def test_pure_python_no_keywords(self):
        """Pure Python without keywords should not match either category."""

        def simple_math(x, y):
            return x + y

        summary = analyze_operations(simple_math)
        # No tensor or orchestration keywords
        assert not summary.has_tensor_ops
        assert not summary.has_orchestration_ops


class TestOperationSummaryProperties:
    """Test OperationSummary computed properties."""

    def test_only_tensor_ops(self):
        """only_tensor_ops should be True when only tensor ops present."""
        summary = OperationSummary(tensor_ops={"jax.numpy"}, orchestration_ops=set())
        assert summary.only_tensor_ops
        assert not summary.only_orchestration_ops
        assert not summary.is_hybrid

    def test_only_orchestration_ops(self):
        """only_orchestration_ops should be True when only orchestration present."""
        summary = OperationSummary(tensor_ops=set(), orchestration_ops={"openai"})
        assert summary.only_orchestration_ops
        assert not summary.only_tensor_ops
        assert not summary.is_hybrid

    def test_is_hybrid(self):
        """is_hybrid should be True when both types present."""
        summary = OperationSummary(tensor_ops={"numpy"}, orchestration_ops={"llm"})
        assert summary.is_hybrid
        assert not summary.only_tensor_ops
        assert not summary.only_orchestration_ops

    def test_neither_type(self):
        """Empty summary should have all False properties."""
        summary = OperationSummary()
        assert not summary.has_tensor_ops
        assert not summary.has_orchestration_ops
        assert not summary.only_tensor_ops
        assert not summary.only_orchestration_ops
        assert not summary.is_hybrid


class TestJaxArrayDetection:
    """Test runtime JAX array detection."""

    def test_plain_list_not_jax_array(self):
        """Plain Python lists should not be detected as JAX arrays."""
        assert not is_jax_array([1, 2, 3])

    def test_plain_dict_not_jax_array(self):
        """Plain Python dicts should not be detected as JAX arrays."""
        assert not is_jax_array({"a": 1, "b": 2})

    def test_none_not_jax_array(self):
        """None should not be detected as JAX array."""
        assert not is_jax_array(None)

    def test_string_not_jax_array(self):
        """Strings should not be detected as JAX arrays."""
        assert not is_jax_array("hello")

    def test_has_jax_arrays_with_plain_args(self):
        """has_jax_arrays should return False for plain Python types."""
        assert not has_jax_arrays([1, 2, 3], {"key": "value"})

    def test_has_jax_arrays_nested_plain(self):
        """has_jax_arrays should handle nested structures."""
        nested = {"data": [1, 2, {"inner": [3, 4]}]}
        assert not has_jax_arrays([nested], {})
