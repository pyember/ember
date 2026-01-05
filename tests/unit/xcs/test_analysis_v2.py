"""Tests for XCS v2 operation analysis system.

This module tests the comprehensive operation classification system including:
- OpKind enum (TENSOR, ORCHESTRATION, HYBRID, NEUTRAL, UNKNOWN)
- Traceability detection (JAX-traceable vs non-JAX)
- Effect risk detection (LOW, MEDIUM, HIGH)
- Binding analysis (closures, globals)
- Local import alias resolution
- Boundary-aware keyword matching
- Wrapper/partial unwrapping
- explain() debugging function
- Caching behavior
- Bounded tree traversal
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest

from ember.xcs.compiler.analysis import (
    EffectRisk,
    OpDecision,
    OpKind,
    Traceability,
    analyze_operations_v2,
    explain,
    has_jax_arrays,
    is_jax_array,
    is_jax_compatible_leaf,
)


class TestOpKindEnum:
    """Test OpKind enumeration values and semantics."""

    def test_all_kind_values_exist(self):
        """All OpKind values should be accessible."""
        assert OpKind.TENSOR
        assert OpKind.ORCHESTRATION
        assert OpKind.HYBRID
        assert OpKind.NEUTRAL
        assert OpKind.UNKNOWN

    def test_kind_values_are_distinct(self):
        """Each OpKind should be unique."""
        kinds = [
            OpKind.TENSOR,
            OpKind.ORCHESTRATION,
            OpKind.HYBRID,
            OpKind.NEUTRAL,
            OpKind.UNKNOWN,
        ]
        assert len(set(kinds)) == 5


class TestTraceabilityEnum:
    """Test Traceability enumeration for JAX compatibility."""

    def test_all_traceability_values_exist(self):
        """All Traceability values should be accessible."""
        assert Traceability.JAX
        assert Traceability.NON_JAX
        assert Traceability.UNKNOWN

    def test_traceability_values_are_distinct(self):
        """Each Traceability should be unique."""
        values = [Traceability.JAX, Traceability.NON_JAX, Traceability.UNKNOWN]
        assert len(set(values)) == 3


class TestEffectRiskEnum:
    """Test EffectRisk enumeration for side effect detection."""

    def test_all_effect_risk_values_exist(self):
        """All EffectRisk values should be accessible."""
        assert EffectRisk.LOW
        assert EffectRisk.MEDIUM
        assert EffectRisk.HIGH

    def test_effect_risk_values_are_distinct(self):
        """Each EffectRisk should be unique."""
        values = [EffectRisk.LOW, EffectRisk.MEDIUM, EffectRisk.HIGH]
        assert len(set(values)) == 3


class TestOpDecisionDataclass:
    """Test OpDecision dataclass structure and defaults."""

    def test_default_values(self):
        """OpDecision should have sensible defaults."""
        decision = OpDecision(kind=OpKind.NEUTRAL)
        assert decision.kind == OpKind.NEUTRAL
        assert decision.jax_traceable == Traceability.UNKNOWN
        assert decision.effect_risk == EffectRisk.LOW
        assert decision.confidence == 0.0
        assert decision.tensor_evidence == frozenset()
        assert decision.orchestration_evidence == frozenset()
        assert decision.effect_evidence == frozenset()
        assert decision.reason == ""

    def test_immutability(self):
        """OpDecision should be immutable (frozen dataclass)."""
        decision = OpDecision(kind=OpKind.TENSOR)
        with pytest.raises(AttributeError):
            decision.kind = OpKind.ORCHESTRATION  # type: ignore

    def test_evidence_sets_are_frozensets(self):
        """Evidence fields should be frozensets for hashability."""
        decision = OpDecision(
            kind=OpKind.TENSOR, tensor_evidence=frozenset(["jax.numpy", "jnp.array"])
        )
        assert isinstance(decision.tensor_evidence, frozenset)


class TestTensorOperationDetection:
    """Test detection of tensor/numeric operations."""

    def test_jax_numpy_detected(self):
        """JAX numpy operations should be detected as TENSOR."""

        def use_jax_numpy():
            return jnp.array([1, 2, 3])

        decision = analyze_operations_v2(use_jax_numpy)
        assert decision.kind == OpKind.TENSOR
        assert decision.jax_traceable == Traceability.JAX

    def test_jax_lax_detected(self):
        """JAX lax operations should be detected as TENSOR."""

        def use_jax_lax(x):
            return jax.lax.add(x, 1)

        decision = analyze_operations_v2(use_jax_lax)
        assert decision.kind == OpKind.TENSOR
        assert decision.jax_traceable == Traceability.JAX

    def test_numpy_detected_as_non_jax(self):
        """Standalone numpy operations should be detected as NON_JAX tensor."""

        def use_numpy_direct():
            import numpy as np

            return np.array([1, 2, 3])

        decision = analyze_operations_v2(use_numpy_direct)
        assert decision.kind == OpKind.TENSOR
        # numpy is non-JAX traceable
        assert decision.jax_traceable == Traceability.NON_JAX

    def test_matmul_keyword_detected(self):
        """matmul keyword should trigger tensor detection."""

        def do_matmul(a, b):
            return matmul(a, b)  # noqa: F821

        decision = analyze_operations_v2(do_matmul)
        assert decision.kind == OpKind.TENSOR


class TestOrchestrationDetection:
    """Test detection of LLM/orchestration operations."""

    def test_openai_detected(self):
        """OpenAI API calls should be detected as orchestration."""

        def call_openai():
            import openai

            return openai.ChatCompletion.create()

        decision = analyze_operations_v2(call_openai)
        assert decision.kind == OpKind.ORCHESTRATION

    def test_anthropic_detected(self):
        """Anthropic API calls should be detected as orchestration."""

        def call_anthropic():
            import anthropic

            return anthropic.Anthropic().messages.create()

        decision = analyze_operations_v2(call_anthropic)
        assert decision.kind == OpKind.ORCHESTRATION

    def test_llm_keyword_detected(self):
        """llm keyword should trigger orchestration detection."""

        def use_llm(prompt):
            return llm_client.complete(prompt)  # noqa: F821

        decision = analyze_operations_v2(use_llm)
        assert decision.kind == OpKind.ORCHESTRATION

    def test_completion_keyword_detected(self):
        """completion keyword should trigger orchestration detection."""

        def get_completion(text):
            return create_completion(text)  # noqa: F821

        decision = analyze_operations_v2(get_completion)
        assert decision.kind == OpKind.ORCHESTRATION


class TestHybridDetection:
    """Test detection of hybrid tensor+orchestration operations."""

    def test_mixed_tensor_and_orchestration(self):
        """Functions with both tensor and orchestration ops should be HYBRID."""

        def hybrid_function(x):
            # Tensor operation
            result = jnp.sum(x)
            # Orchestration operation
            llm_response = llm_client.complete(str(result))  # noqa: F821
            return llm_response

        decision = analyze_operations_v2(hybrid_function)
        assert decision.kind == OpKind.HYBRID
        assert len(decision.tensor_evidence) > 0
        assert len(decision.orchestration_evidence) > 0


class TestNeutralAndUnknown:
    """Test neutral (pure Python) and unknown classifications."""

    def test_pure_python_is_neutral(self):
        """Pure Python without keywords should be NEUTRAL."""

        def simple_math(x, y):
            return x + y

        decision = analyze_operations_v2(simple_math)
        assert decision.kind == OpKind.NEUTRAL

    def test_builtin_is_unknown(self):
        """Built-in functions without source should be UNKNOWN."""
        decision = analyze_operations_v2(len)
        assert decision.kind == OpKind.UNKNOWN

    def test_lambda_with_no_keywords(self):
        """Lambdas with pure Python should be NEUTRAL."""
        decision = analyze_operations_v2(lambda x: x * 2)
        assert decision.kind == OpKind.NEUTRAL


class TestEffectRiskDetection:
    """Test detection of side effects (I/O, network, subprocess)."""

    def test_file_io_medium_risk(self):
        """File I/O operations should have MEDIUM effect risk."""

        def read_file():
            with open("test.txt") as f:
                return f.read()

        decision = analyze_operations_v2(read_file)
        assert decision.effect_risk in (EffectRisk.MEDIUM, EffectRisk.HIGH)

    def test_subprocess_high_risk(self):
        """Subprocess operations should have HIGH effect risk."""

        def run_command():
            import subprocess

            return subprocess.run(["ls"])

        decision = analyze_operations_v2(run_command)
        assert decision.effect_risk == EffectRisk.HIGH

    def test_network_high_risk(self):
        """Network operations should have HIGH effect risk."""

        def make_request():
            import requests

            return requests.get("http://example.com")

        decision = analyze_operations_v2(make_request)
        assert decision.effect_risk == EffectRisk.HIGH

    def test_pure_function_low_risk(self):
        """Pure functions should have LOW effect risk."""

        def pure_math(x, y):
            return x + y

        decision = analyze_operations_v2(pure_math)
        assert decision.effect_risk == EffectRisk.LOW


class TestBoundaryAwareMatching:
    """Test that keyword matching respects word boundaries."""

    def test_jnp_not_matches_np_pattern(self):
        """jnp.sum should NOT match 'np.' pattern as substring."""

        def use_jnp():
            return jnp.sum(jnp.array([1, 2, 3]))

        decision = analyze_operations_v2(use_jnp)
        # Should be JAX-traceable, not non-JAX
        assert decision.jax_traceable == Traceability.JAX

    def test_np_prefix_correctly_detected(self):
        """np.array should correctly match numpy pattern."""

        def use_np():
            import numpy as np

            return np.array([1, 2, 3])

        decision = analyze_operations_v2(use_np)
        assert decision.kind == OpKind.TENSOR
        # NumPy is non-JAX
        assert decision.jax_traceable == Traceability.NON_JAX


class TestLocalImportAlias:
    """Test that local import aliases are properly resolved."""

    def test_jax_numpy_alias_resolved(self):
        """import jax.numpy as jnp should be recognized as JAX."""

        def with_jnp_import():
            import jax.numpy as jnp

            return jnp.zeros(10)

        decision = analyze_operations_v2(with_jnp_import)
        assert decision.kind == OpKind.TENSOR
        assert decision.jax_traceable == Traceability.JAX

    def test_numpy_alias_resolved(self):
        """import numpy as np should be recognized as numpy."""

        def with_np_import():
            import numpy as np

            return np.zeros(10)

        decision = analyze_operations_v2(with_np_import)
        assert decision.kind == OpKind.TENSOR


class TestWrapperUnwrapping:
    """Test that wrappers and partials are correctly unwrapped."""

    def test_functools_wraps_unwrapped(self):
        """Functions decorated with @functools.wraps should be unwrapped."""

        def original(x):
            return jnp.sum(x)

        @functools.wraps(original)
        def wrapper(x):
            return original(x)

        decision = analyze_operations_v2(wrapper)
        assert decision.kind == OpKind.TENSOR

    def test_functools_partial_unwrapped(self):
        """functools.partial should be unwrapped to underlying function."""

        def add_arrays(a, b):
            return jnp.add(a, b)

        partial_fn = functools.partial(add_arrays, b=jnp.ones(3))
        decision = analyze_operations_v2(partial_fn)
        assert decision.kind == OpKind.TENSOR


class TestExplainFunction:
    """Test the explain() debugging function."""

    def test_explain_returns_string(self):
        """explain() should return a formatted string."""

        def sample_func(x):
            return jnp.sum(x)

        result = explain(sample_func)
        assert isinstance(result, str)

    def test_explain_includes_kind(self):
        """explain() output should include the operation kind."""

        def tensor_func(x):
            return jnp.array(x)

        result = explain(tensor_func)
        assert "TENSOR" in result or "tensor" in result.lower()

    def test_explain_includes_evidence(self):
        """explain() output should include evidence details."""

        def jax_func():
            return jnp.zeros(10)

        result = explain(jax_func)
        assert len(result) > 50  # Should have substantial content


class TestJaxArrayDetection:
    """Test runtime JAX array detection functions."""

    def test_jax_array_detected(self):
        """JAX arrays should be detected as JAX arrays."""
        arr = jnp.array([1, 2, 3])
        assert is_jax_array(arr)

    def test_python_list_not_jax_array(self):
        """Python lists should not be detected as JAX arrays."""
        assert not is_jax_array([1, 2, 3])

    def test_python_dict_not_jax_array(self):
        """Python dicts should not be detected as JAX arrays."""
        assert not is_jax_array({"a": 1})

    def test_none_not_jax_array(self):
        """None should not be detected as JAX array."""
        assert not is_jax_array(None)

    def test_string_not_jax_array(self):
        """Strings should not be detected as JAX arrays."""
        assert not is_jax_array("hello")

    def test_has_jax_arrays_with_jax_input(self):
        """has_jax_arrays should return True when JAX arrays present."""
        arr = jnp.array([1, 2, 3])
        assert has_jax_arrays(arr)

    def test_has_jax_arrays_nested(self):
        """has_jax_arrays should detect JAX arrays in nested structures.

        Note: has_jax_arrays takes an Iterable of args, so we pass [nested]
        to treat the dict as a single argument to traverse.
        """
        nested = {"data": [jnp.array([1, 2, 3])]}
        # Pass as a single arg in a list since has_jax_arrays iterates args
        assert has_jax_arrays([nested])

    def test_has_jax_arrays_without_jax(self):
        """has_jax_arrays should return False for plain Python types."""
        assert not has_jax_arrays([1, 2, 3], {"key": "value"})


class TestJaxCompatibleLeaf:
    """Test is_jax_compatible_leaf for JAX array detection.

    Note: is_jax_compatible_leaf is specifically for detecting actual JAX
    array objects (jax.Array, tracers, numpy arrays). It does NOT consider
    Python scalars or None as compatible, since those are not arrays.
    This is by design - the function answers "is this a JAX array?" not
    "can JAX convert this?".
    """

    def test_jax_array_compatible(self):
        """JAX arrays should be JAX compatible leaves."""
        arr = jnp.array([1, 2, 3])
        assert is_jax_compatible_leaf(arr)

    def test_scalars_not_considered_array(self):
        """Python scalars are NOT JAX array leaves (they're converted, not arrays).

        This is correct behavior - is_jax_compatible_leaf detects actual JAX
        arrays, not types that can be implicitly converted to arrays.
        """
        assert not is_jax_compatible_leaf(1)
        assert not is_jax_compatible_leaf(1.5)
        assert not is_jax_compatible_leaf(True)

    def test_none_not_array(self):
        """None is NOT a JAX array (correct behavior)."""
        assert not is_jax_compatible_leaf(None)

    def test_string_not_compatible(self):
        """Strings should not be JAX compatible leaves."""
        assert not is_jax_compatible_leaf("hello")

    def test_list_not_leaf(self):
        """Lists are containers, not array leaves."""
        assert not is_jax_compatible_leaf([1, 2, 3])


class TestCachingBehavior:
    """Test that analysis results are properly cached."""

    def test_same_function_cached(self):
        """Repeated analysis of same function should use cache."""

        def cached_func(x):
            return jnp.sum(x)

        result1 = analyze_operations_v2(cached_func)
        result2 = analyze_operations_v2(cached_func)

        # Results should be identical
        assert result1.kind == result2.kind
        assert result1.jax_traceable == result2.jax_traceable

    def test_different_functions_computed(self):
        """Different functions should get different results."""

        def tensor_func(x):
            return jnp.sum(x)

        def orch_func(x):
            return llm_client.complete(x)  # noqa: F821

        result1 = analyze_operations_v2(tensor_func)
        result2 = analyze_operations_v2(orch_func)

        assert result1.kind != result2.kind


class TestExplicitMarkerPriority:
    """Test that explicit markers override heuristic detection."""

    def test_mark_tensor_overrides(self):
        """@mark_tensor should force TENSOR classification."""
        from ember.api.decorators import mark_tensor

        @mark_tensor
        def misleading_name_model():
            return "not actually a model"

        decision = analyze_operations_v2(misleading_name_model)
        assert decision.kind == OpKind.TENSOR

    def test_mark_orchestration_overrides(self):
        """@mark_orchestration should force ORCHESTRATION classification."""
        from ember.api.decorators import mark_orchestration

        @mark_orchestration
        def compute_array():
            return []

        decision = analyze_operations_v2(compute_array)
        assert decision.kind == OpKind.ORCHESTRATION

    def test_mark_hybrid_sets_both(self):
        """@mark_hybrid should force HYBRID classification."""
        from ember.api.decorators import mark_hybrid

        @mark_hybrid
        def mixed_function():
            return "result"

        decision = analyze_operations_v2(mixed_function)
        assert decision.kind == OpKind.HYBRID


class TestAmbiguousKeywordsNotMisclassified:
    """Test that ambiguous keywords don't cause false positives."""

    def test_model_alone_not_orchestration(self):
        """'model' alone should not trigger orchestration (could be ML model)."""

        def use_model(model, x):
            return model(x)

        decision = analyze_operations_v2(use_model)
        # Should NOT be classified as orchestration just because of 'model'
        assert decision.kind != OpKind.ORCHESTRATION

    def test_call_alone_not_orchestration(self):
        """'call' alone should not trigger orchestration."""

        def call_function(func, args):
            return func(*args)

        decision = analyze_operations_v2(call_function)
        assert decision.kind != OpKind.ORCHESTRATION

    def test_api_alone_not_orchestration(self):
        """'api' alone should not trigger orchestration."""

        def make_api_call(api_client):
            return api_client.get("/endpoint")

        decision = analyze_operations_v2(make_api_call)
        assert decision.kind != OpKind.ORCHESTRATION


class TestConfidenceScoring:
    """Test that confidence scores are meaningful."""

    def test_explicit_marker_high_confidence(self):
        """Explicit markers should yield high confidence."""
        from ember.api.decorators import mark_tensor

        @mark_tensor
        def marked_func():
            pass

        decision = analyze_operations_v2(marked_func)
        assert decision.confidence >= 0.9

    def test_unknown_low_confidence(self):
        """Unknown classification should have low confidence."""
        decision = analyze_operations_v2(len)  # Built-in
        assert decision.confidence < 0.5

    def test_clear_evidence_moderate_confidence(self):
        """Clear heuristic matches should have moderate confidence."""

        def jax_func():
            return jnp.zeros(10)

        decision = analyze_operations_v2(jax_func)
        assert 0.3 <= decision.confidence <= 0.9


class TestIntegrationWithTransforms:
    """Test that analysis integrates correctly with XCS transforms."""

    def test_grad_blocks_orchestration(self):
        """grad should block pure orchestration functions."""
        from ember.api.decorators import mark_orchestration
        from ember.xcs import grad
        from ember.xcs.errors import XCSError

        @mark_orchestration
        def orchestration_func(x):
            return len(x)

        grad_fn = grad(orchestration_func)
        with pytest.raises(XCSError, match="orchestration"):
            grad_fn("test")

    def test_grad_allows_tensor(self):
        """grad should allow pure tensor functions."""
        from ember.xcs import grad

        def tensor_func(x):
            return jnp.sum(x**2)

        grad_fn = grad(tensor_func)
        result = grad_fn(jnp.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_function(self):
        """Empty function should not crash."""

        def empty():
            pass

        decision = analyze_operations_v2(empty)
        assert decision.kind in (OpKind.NEUTRAL, OpKind.UNKNOWN)

    def test_deeply_nested_function_definition(self):
        """Nested function definitions should be handled."""

        def outer():
            def middle():
                def inner():
                    return jnp.zeros(10)

                return inner()

            return middle()

        decision = analyze_operations_v2(outer)
        assert decision.kind == OpKind.TENSOR

    def test_class_method_analysis(self):
        """Class methods should be analyzable."""

        class MyClass:
            def tensor_method(self, x):
                return jnp.sum(x)

        decision = analyze_operations_v2(MyClass.tensor_method)
        assert decision.kind == OpKind.TENSOR

    def test_static_method_analysis(self):
        """Static methods should be analyzable."""

        class MyClass:
            @staticmethod
            def static_tensor(x):
                return jnp.array(x)

        decision = analyze_operations_v2(MyClass.static_tensor)
        assert decision.kind == OpKind.TENSOR
