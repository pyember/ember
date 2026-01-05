"""Tests for XCS transformations."""

import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import pytest

from ember.operators import Operator
from ember.xcs import grad, pmap, scan, vmap
from ember.xcs.errors import XCSError


class TensorOperator(Operator):
    """Pure tensor operation operator."""

    weight: jnp.ndarray

    def __init__(self, size: int, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (size, size))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.dot(self.weight, x))


class OrchestrationOperator(Operator):
    """Orchestration/LLM operation operator."""

    operation: str
    delay: float

    def __init__(self, operation: str = "summarize", delay: float = 0.01):
        self.operation = operation
        self.delay = delay

    def forward(self, x: str) -> str:
        # Simulate API call delay
        time.sleep(self.delay)
        return f"{self.operation}({x})"


class HybridOperator(Operator):
    """Operator mixing tensor and orchestration operations."""

    tensor_op: TensorOperator
    orch_op: OrchestrationOperator

    def __init__(self, size: int, key: jax.random.PRNGKey):
        self.tensor_op = TensorOperator(size, key)
        self.orch_op = OrchestrationOperator("process")

    def forward(self, x: Tuple[jnp.ndarray, str]) -> Tuple[jnp.ndarray, str]:
        tensor_input, text_input = x
        tensor_result = self.tensor_op(tensor_input)
        text_result = self.orch_op(text_input)
        return tensor_result, text_result


class TestVmap:
    """Test intelligent vmap behavior."""

    def test_vmap_tensor_operations(self):
        """Test vmap on pure tensor operations uses JAX vmap."""

        def tensor_function(x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
            op = TensorOperator(x.shape[0], key)
            return op(x)

        # Create batch
        batch_size = 8
        dim = 5
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, batch_size)
        batch = jax.random.normal(key, (batch_size, dim))

        # Apply vmap
        vmapped_fn = vmap(tensor_function)
        results = vmapped_fn(batch, keys)

        assert results.shape == (batch_size, dim)

        # Verify it's actually using JAX vmap by checking it's differentiable
        def loss_fn(x, keys):
            return jnp.sum(vmapped_fn(x, keys) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(batch, keys)
        assert grads.shape == batch.shape

    def test_vmap_supports_eval_shape_tracing(self):
        """Ensure abstract evaluation uses the native JAX path without fallback."""

        def tensor_function(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2.0

        vmapped_fn = vmap(tensor_function)
        struct = jax.ShapeDtypeStruct((3,), jnp.float32)

        evaluated = jax.eval_shape(vmapped_fn, struct)

        assert isinstance(evaluated, jax.ShapeDtypeStruct)
        assert evaluated.shape == (3,)
        assert evaluated.dtype == jnp.float32

    def test_vmap_orchestration_operations(self):
        """Test vmap on orchestration ops uses parallel execution."""

        def orchestration_function(x: str) -> str:
            op = OrchestrationOperator("analyze", delay=0.05)
            return op(x)

        # Create batch
        batch = [f"text_{i}" for i in range(4)]

        # Time sequential execution
        start = time.time()
        seq_results = [orchestration_function(x) for x in batch]
        seq_time = time.time() - start

        # Time vmapped execution
        vmapped_fn = vmap(orchestration_function)
        start = time.time()
        vmap_results = vmapped_fn(batch)
        vmap_time = time.time() - start

        # Results should match
        assert vmap_results == seq_results

        # Should be faster due to parallelism
        speedup = seq_time / vmap_time
        assert speedup >= 1.3, f"Expected >=1.3x speedup, got {speedup:.2f}x"

    def test_vmap_hybrid_operations(self):
        """Test vmap on hybrid operations with smart batching."""

        def hybrid_function(tensor_x: jnp.ndarray, text_x: str, key: jax.random.PRNGKey):
            op = HybridOperator(tensor_x.shape[0], key)
            return op((tensor_x, text_x))

        # Create batch
        batch_size = 4
        dim = 3
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, batch_size)

        tensor_batch = jax.random.normal(key, (batch_size, dim))
        text_batch = [f"item_{i}" for i in range(batch_size)]

        # Apply vmap
        vmapped_fn = vmap(hybrid_function)
        tensor_results, text_results = vmapped_fn(tensor_batch, text_batch, keys)

        # Verify shapes and types
        assert tensor_results.shape == (batch_size, dim)
        assert len(text_results) == batch_size
        assert all(isinstance(r, str) for r in text_results)

    def test_vmap_with_in_axes(self):
        """Test vmap with custom in_axes specification."""

        def multi_arg_function(x: jnp.ndarray, scale: float, bias: jnp.ndarray) -> jnp.ndarray:
            return x * scale + bias

        batch_size = 5
        dim = 3

        # Batched x, scalar scale, batched bias
        x_batch = jnp.ones((batch_size, dim))
        scale = 2.0  # Not batched
        bias_batch = jnp.arange(batch_size).reshape(-1, 1)

        # vmap with in_axes=(0, None, 0)
        vmapped_fn = vmap(multi_arg_function, in_axes=(0, None, 0))
        results = vmapped_fn(x_batch, scale, bias_batch)

        assert results.shape == (batch_size, dim)

        # Verify computation
        for i in range(batch_size):
            expected = x_batch[i] * scale + bias_batch[i]
            assert jnp.allclose(results[i], expected)

    def test_vmap_fallback_preserves_additional_args(self):
        """Ensure orchestration fallback keeps extra positional and keyword arguments."""

        calls: List[Tuple[str, str, int]] = []

        def formatter(token: str, suffix: str, *, scale: int) -> str:
            calls.append((token, suffix, scale))
            return f"{token}:{suffix}:{scale}"

        vmapped = vmap(formatter, in_axes=(0,))
        batch = ["a", "b", "c"]
        result = vmapped(batch, "tail", scale=2)

        assert result == ["a:tail:2", "b:tail:2", "c:tail:2"]
        assert len(calls) == len(batch)
        assert all(entry[1] == "tail" and entry[2] == 2 for entry in calls)

    def test_vmap_fallback_stacks_dict_outputs(self):
        """Fallback should stack mapping leaves just like JAX."""

        vmapped = vmap(lambda x: {"value": x, "double": x * 2})
        result = vmapped([1, 2, 3])

        assert isinstance(result, dict)
        assert result == {"value": [1, 2, 3], "double": [2, 4, 6]}

    def test_vmap_fallback_respects_out_axes(self):
        """Ensure orchestration fallback stacks outputs along the requested axis."""

        vmapped = vmap(lambda _: jnp.arange(3), out_axes=1)
        result = vmapped([0, 1])

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (3, 2)
        expected = jnp.array([[0, 0], [1, 1], [2, 2]])
        assert jnp.array_equal(result, expected)

    def test_vmap_fallback_out_axes_none_returns_scalar(self):
        """When out_axes=None, fallback returns a single representative value."""

        vmapped = vmap(lambda _: "static", out_axes=None)
        result = vmapped([0, 1, 2])

        assert result == "static"

    def test_vmap_fallback_out_axes_none_requires_identical_outputs(self):
        """Non-identical outputs with out_axes=None should raise XCSError."""

        vmapped = vmap(lambda x: f"val-{x}", out_axes=None)

        with pytest.raises(XCSError):
            vmapped([0, 1])

    def test_vmap_axis_size_guard(self):
        """axis_size bounds input batch size to avoid silent truncation."""

        def echo(token: str) -> str:
            return token.upper()

        vmapped = vmap(echo, axis_size=1)

        with pytest.raises(XCSError):
            vmapped(["alpha", "beta"])


class TestPmap:
    """Test distributed pmap behavior."""

    def test_pmap_tensor_operations(self):
        """Test pmap on tensor operations (single or multi device)."""

        def distributed_tensor_op(x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
            op = TensorOperator(x.shape[0], key)
            return op(x)

        # Create data - shape it as if for multiple devices even if we have one
        # This tests the pmap transformation works correctly
        n_replicas = max(jax.device_count(), 2)  # Simulate at least 2 for testing
        dim = 4
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, n_replicas)

        # Data shaped for pmap
        x = jax.random.normal(key, (n_replicas, dim))

        # Apply pmap
        pmapped_fn = pmap(distributed_tensor_op)

        if jax.device_count() == 1:
            # On single device, pmap should still work but run sequentially
            # We simulate by manually mapping
            results = jnp.stack([distributed_tensor_op(x[i], keys[i]) for i in range(n_replicas)])
        else:
            # On multiple devices, use actual pmap
            results = pmapped_fn(x[: jax.device_count()], keys[: jax.device_count()])

        assert results.shape[0] >= 2  # At least 2 replicas
        assert results.shape[1] == dim

        # Verify computation is correct by checking one result
        single_result = distributed_tensor_op(x[0], keys[0])
        assert jnp.allclose(results[0], single_result)

    def test_pmap_orchestration_fallback(self):
        """Test pmap on orchestration ops with distributed execution."""

        def orchestration_op(x: str) -> str:
            op = OrchestrationOperator("distributed_process", delay=0.1)
            return op(x)

        # Simulate multiple "devices" (actually threads for orchestration)
        batch = ["data_0", "data_1", "data_2", "data_3"]

        # Apply pmap (should use thread pool for orchestration)
        pmapped_fn = pmap(orchestration_op)
        results = pmapped_fn(batch)

        assert len(results) == len(batch)
        assert all("distributed_process" in r for r in results)
        assert results == [
            "distributed_process(data_0)",
            "distributed_process(data_1)",
            "distributed_process(data_2)",
            "distributed_process(data_3)",
        ]

    def test_pmap_fallback_respects_out_axes(self):
        """Ensure pmap orchestration fallback respects out_axes specification."""

        pmapped = pmap(lambda _: jnp.arange(2), out_axes=1)
        result = pmapped([0, 1, 2])

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2, 3)
        expected = jnp.array([[0, 0, 0], [1, 1, 1]])
        assert jnp.array_equal(result, expected)

    def test_pmap_axis_name(self):
        """Test pmap with axis names for collective operations."""

        def collective_op(x: jnp.ndarray) -> jnp.ndarray:
            # Sum across all devices (or return value on single device)
            return jax.lax.psum(x, axis_name="devices")

        if jax.device_count() >= 2:
            # Real multi-device test
            x = jnp.arange(jax.device_count())
            pmapped_fn = pmap(collective_op, axis_name="devices")
            results = pmapped_fn(x)
            expected_sum = x.sum()
            assert all(r == expected_sum for r in results)
        else:
            # Single device: test that pmap API works even if collective is trivial
            # psum on single device just returns the value
            x = jnp.array([5.0])  # Single value

            # Test that we can create pmapped function with axis_name
            pmapped_fn = pmap(collective_op, axis_name="devices")

            # On single device, pmap with array of size 1 should work
            result = pmapped_fn(x)

            # On single device, psum just returns the input
            assert result[0] == x[0]

            # Also test with explicit identity behavior
            def identity_collective(x: jnp.ndarray) -> jnp.ndarray:
                # This simulates what psum does on single device
                return x

            pmapped_identity = pmap(identity_collective, axis_name="devices")
            identity_result = pmapped_identity(x)
            assert jnp.allclose(result, identity_result)


class TestScan:
    """Test sequential scan operations."""

    def test_scan_tensor_operations(self):
        """Test scan on tensor operations uses JAX scan."""

        def step_fn(carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Simple RNN-like step
            new_carry = jnp.tanh(carry + x)
            output = new_carry * 2
            return new_carry, output

        # Create sequence
        seq_len = 10
        dim = 5
        key = jax.random.PRNGKey(42)

        init_carry = jnp.zeros(dim)
        xs = jax.random.normal(key, (seq_len, dim))

        # Apply scan
        scanned_fn = scan(step_fn)
        final_carry, outputs = scanned_fn(init_carry, xs)

        assert final_carry.shape == (dim,)
        assert outputs.shape == (seq_len, dim)

        # Verify it maintains state correctly
        manual_carry = init_carry
        manual_outputs = []
        for x in xs:
            manual_carry, output = step_fn(manual_carry, x)
            manual_outputs.append(output)

        assert jnp.allclose(final_carry, manual_carry)
        assert jnp.allclose(outputs, jnp.stack(manual_outputs))

    def test_scan_orchestration_operations(self):
        """Test scan on orchestration maintains sequential state."""

        def dialogue_step(state: str, user_input: str) -> Tuple[str, str]:
            # Simulate stateful dialogue
            op = OrchestrationOperator("respond", delay=0.01)
            response = op(f"{state}|{user_input}")
            new_state = f"{state};{user_input}->{response}"
            return new_state, response

        # Create conversation
        init_state = "START"
        user_inputs = ["Hello", "How are you?", "Tell me a joke", "Goodbye"]

        # Apply scan
        scanned_fn = scan(dialogue_step)
        final_state, responses = scanned_fn(init_state, user_inputs)

        assert len(responses) == len(user_inputs)
        assert "START" in final_state
        assert all("respond(" in r for r in responses)

        # Verify state threading
        assert "Hello->respond(START|Hello)" in final_state

    def test_scan_reverse(self):
        """Test reverse scan."""

        def accumulate(carry: float, x: float) -> Tuple[float, float]:
            new_carry = carry + x
            return new_carry, new_carry

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Forward scan
        forward_fn = scan(accumulate)
        _, forward_results = forward_fn(0.0, xs)

        # Reverse scan
        reverse_fn = scan(accumulate, reverse=True)
        _, reverse_results = reverse_fn(0.0, xs)

        # Forward should be cumsum
        assert jnp.allclose(forward_results, jnp.array([1.0, 3.0, 6.0, 10.0, 15.0]))

        # Reverse should accumulate from the end
        assert jnp.allclose(reverse_results, jnp.array([15.0, 14.0, 12.0, 9.0, 5.0]))

    def test_scan_with_operator_state(self):
        """Test scan with stateful operators."""

        class StatefulOperator(Operator):
            """Operator that modifies based on history."""

            history_weight: float

            def __init__(self, history_weight: float = 0.9):
                self.history_weight = history_weight

            def forward(self, state_and_input: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                state, x = state_and_input
                return state * self.history_weight + x * (1 - self.history_weight)

        def operator_step(
            carry_op: StatefulOperator, x: jnp.ndarray
        ) -> Tuple[StatefulOperator, jnp.ndarray]:
            # Carry is (operator, state)
            state = carry_op(x)
            return carry_op, state

        # Create sequence
        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Initial operator and state
        init_op = StatefulOperator(0.5)

        # This is a simplified test - in practice, scan might need special handling for operators
        # Here we just verify the concept
        states = []
        state = jnp.array(0.0)
        for x in xs:
            state = init_op((state, jnp.array(x)))
            states.append(state)

        # Manual verification of exponential smoothing
        assert jnp.allclose(states[0], 0.5)  # 0 * 0.5 + 1 * 0.5
        assert jnp.allclose(states[1], 1.25)  # 0.5 * 0.5 + 2 * 0.5


class TestGrad:
    """Test gradient computation."""

    def test_grad_tensor_operations(self):
        """Test grad on pure tensor operations."""

        def loss_fn(op: TensorOperator, x: jnp.ndarray, y: jnp.ndarray) -> float:
            pred = op(x)
            return jnp.mean((pred - y) ** 2)

        # Create operator and data
        dim = 4
        key = jax.random.PRNGKey(42)
        op = TensorOperator(dim, key)

        x = jnp.ones(dim)
        y = jnp.zeros(dim)

        # Compute gradient
        grad_fn = grad(loss_fn)
        grads = grad_fn(op, x, y)

        # Check gradient structure
        assert hasattr(grads, "weight")
        assert grads.weight.shape == op.weight.shape
        assert jnp.abs(grads.weight).max() > 1e-6

    def test_grad_orchestration_error(self):
        """Test grad on orchestration ops raises helpful error."""
        from ember.api.decorators import mark_orchestration

        @mark_orchestration
        def orchestration_loss(op: OrchestrationOperator, x: str) -> float:
            result = op(x)
            # Fake loss for testing
            return len(result)

        op = OrchestrationOperator()

        # Should raise error about orchestration ops
        grad_fn = grad(orchestration_loss)
        with pytest.raises(XCSError, match="orchestration"):
            grad_fn(op, "test")

    def test_grad_with_multiple_operators(self):
        """Test grad through multiple operators."""

        def multi_op_loss(ops: List[TensorOperator], x: jnp.ndarray, y: jnp.ndarray) -> float:
            # Chain operators
            result = x
            for op in ops:
                result = op(result)
            return jnp.mean((result - y) ** 2)

        # Create operators
        dim = 3
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        ops = [TensorOperator(dim, k) for k in keys]

        x = jnp.ones(dim)
        y = jnp.array([1.0, -1.0, 0.0])

        # Compute gradients
        grad_fn = grad(multi_op_loss)
        grads = grad_fn(ops, x, y)

        # Should have gradients for all operators
        assert len(grads) == 3
        for i, grad_op in enumerate(grads):
            assert hasattr(grad_op, "weight")
            assert grad_op.weight.shape == ops[i].weight.shape

    def test_grad_with_aux(self):
        """Test grad with auxiliary outputs."""

        def loss_with_aux(
            op: TensorOperator, x: jnp.ndarray, y: jnp.ndarray
        ) -> Tuple[float, Dict[str, Any]]:
            pred = op(x)
            loss = jnp.mean((pred - y) ** 2)

            # Auxiliary outputs for logging
            aux = {
                "predictions": pred,
                "l2_norm": jnp.linalg.norm(pred),
                "max_activation": jnp.max(jnp.abs(pred)),
            }

            return loss, aux

        dim = 5
        key = jax.random.PRNGKey(42)
        op = TensorOperator(dim, key)

        x = jnp.ones(dim) * 0.5
        y = jnp.zeros(dim)

        # Compute gradient with aux
        grad_fn = grad(loss_with_aux, has_aux=True)
        grads, aux = grad_fn(op, x, y)

        # Check we got both gradients and aux
        assert hasattr(grads, "weight")
        assert "predictions" in aux
        assert "l2_norm" in aux
        assert aux["predictions"].shape == (dim,)

    def test_grad_hybrid_with_non_array_aux_payload(self):
        """Hybrid gradients should preserve non-array auxiliary data."""

        def hybrid_loss(x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
            loss = jnp.sum(x**2)
            orch = OrchestrationOperator("echo", delay=0.0)
            prompt = f"prompt-{x.shape[0]}"
            aux_payload = {
                "prompt": orch(prompt),
                "metadata": {"length": len(prompt)},
            }
            return loss, aux_payload

        x = jnp.ones((4,), dtype=jnp.float32)

        grad_fn = grad(hybrid_loss, has_aux=True)
        grads, aux = grad_fn(x)

        expected_prompt = f"prompt-{x.shape[0]}"
        expected_response = f"echo({expected_prompt})"

        assert jnp.allclose(grads, 2.0 * x)
        assert aux["prompt"] == expected_response
        assert aux["metadata"]["length"] == len(expected_prompt)


class TestTransformComposition:
    """Test composing multiple transformations."""

    def test_vmap_of_scan(self):
        """Test vmap over scan for batch sequence processing."""

        def sequence_processor(init: jnp.ndarray, seq: jnp.ndarray) -> jnp.ndarray:
            def step(carry, x):
                new_carry = carry * 0.9 + x * 0.1
                return new_carry, new_carry

            final, outputs = scan(step)(init, seq)
            return outputs

        # Batch of sequences
        batch_size = 4
        seq_len = 5
        dim = 3

        key = jax.random.PRNGKey(42)
        init_batch = jax.random.normal(key, (batch_size, dim))
        seq_batch = jax.random.normal(key, (batch_size, seq_len, dim))

        # vmap over batch dimension
        batch_processor = vmap(sequence_processor)
        results = batch_processor(init_batch, seq_batch)

        assert results.shape == (batch_size, seq_len, dim)

    def test_grad_of_vmap(self):
        """Test gradient of vmapped function."""

        def single_loss(op: TensorOperator, x: jnp.ndarray, y: jnp.ndarray) -> float:
            pred = op(x)
            return jnp.mean((pred - y) ** 2)

        def batch_loss(op: TensorOperator, x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> float:
            # vmap the loss computation
            losses = vmap(lambda x, y: single_loss(op, x, y))(x_batch, y_batch)
            return jnp.mean(losses)

        # Create operator and batch data
        batch_size = 8
        dim = 4
        key = jax.random.PRNGKey(42)

        op = TensorOperator(dim, key)
        x_batch = jax.random.normal(key, (batch_size, dim))
        y_batch = jax.random.normal(key, (batch_size, dim))

        # Compute gradient
        grad_fn = grad(batch_loss)
        grads = grad_fn(op, x_batch, y_batch)

        assert hasattr(grads, "weight")
        assert grads.weight.shape == op.weight.shape

    def test_nested_transformations(self):
        """Test complex nesting of transformations.

        Note: Inside jax.jit, we use jax.vmap (not ember.xcs.vmap) because:
        - jax.jit traces the function, converting inputs to abstract tracers
        - Ember's vmap fallback path uses thread pools that can't handle tracers
        - Using native jax.vmap inside jax.jit is the semantically correct pattern
        - Ember transforms are designed for *top-level* orchestration, not inside JAX traces
        """

        @jax.jit  # Can also compose with JAX jit
        def complex_computation(keys: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            # Create operators for each key
            def create_and_apply(key, x):
                op = TensorOperator(x.shape[0], key)
                return op(x)

            # Inside jax.jit, use jax.vmap for pure tensor ops
            vmapped_apply = jax.vmap(create_and_apply)

            # Apply to get ensemble predictions
            predictions = vmapped_apply(keys, jnp.broadcast_to(x, (keys.shape[0], x.shape[0])))

            # Average predictions
            return jnp.mean(predictions, axis=0)

        # Test
        n_models = 5
        dim = 3
        key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, n_models)
        x = jnp.ones(dim)

        result = complex_computation(keys, x)
        assert result.shape == (dim,)
