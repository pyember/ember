"""Tests for EmberContext."""

import asyncio
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from ember.context import (
    EmberContext,
    context,
    get_config,
    set_config,
)
from ember.core.credentials import CredentialNotFoundError

create_context = context.manager
get_context = context.get
with_context = context.manager


@pytest.fixture(autouse=True)
def reset_context():
    """Reset context between tests to ensure isolation."""
    # Clear any existing context
    if hasattr(EmberContext._thread_local, "context"):
        delattr(EmberContext._thread_local, "context")
    EmberContext._context_var.set(None)
    yield
    # Clean up after test
    if hasattr(EmberContext._thread_local, "context"):
        delattr(EmberContext._thread_local, "context")
    EmberContext._context_var.set(None)


class TestContextCore:
    """Core context functionality."""

    def test_singleton(self, reset_context):
        """Singleton returns same instance."""
        assert EmberContext.current() is EmberContext.current()

    def test_isolation(self, reset_context, tmp_path):
        """Isolated contexts are independent."""
        # Create two isolated contexts with different config paths
        ctx1 = EmberContext(isolated=True)
        ctx1._config_file = tmp_path / "ctx1" / "config.yaml"
        ctx1._config = {}  # Start with empty config

        ctx2 = EmberContext(isolated=True)
        ctx2._config_file = tmp_path / "ctx2" / "config.yaml"
        ctx2._config = {}  # Start with empty config

        # Contexts should be different instances
        assert ctx1 != ctx2

        # Changes to ctx1 don't affect ctx2
        ctx1.set_config("test", "value1")
        assert ctx2.get_config("test") is None

        # And vice versa
        ctx2.set_config("test", "value2")
        assert ctx1.get_config("test") == "value1"
        assert ctx2.get_config("test") == "value2"

    def test_config_operations(self):
        """Config get/set with dot notation."""
        ctx = EmberContext(isolated=True)

        # Nested set/get
        ctx.set_config("a.b.c", "value")
        assert ctx.get_config("a.b.c") == "value"
        assert ctx.get_config("a.b") == {"c": "value"}

        # Default values
        assert ctx.get_config("missing", "default") == "default"

        # Edge cases
        assert ctx.get_config("") is None
        assert ctx.get_config(None) is None

    def test_performance(self):
        """Singleton access is fast (constant time, not O(n))."""
        EmberContext.current()  # Initialize

        start = time.perf_counter()
        for _ in range(10000):
            EmberContext.current()
        elapsed = time.perf_counter() - start

        # Should be fast (< 100ms for 10k accesses = < 10μs per access)
        # This validates O(1) access, not absolute performance
        assert elapsed < 0.1, f"Singleton access too slow: {elapsed*1000:.1f}ms for 10k accesses"


class TestContextInheritance:
    """Context inheritance and isolation."""

    def test_child_isolation(self):
        """Child contexts inherit but don't affect parent."""
        parent = EmberContext(isolated=True)
        parent.set_config("shared", "parent")
        parent.set_config("override", "parent")

        child = parent.create_child(override="child")

        # Child inherits
        assert child.get_config("shared") == "parent"

        # Child overrides
        assert child.get_config("override") == "child"

        # Parent unchanged
        assert parent.get_config("override") == "parent"

        # Child changes don't propagate
        child.set_config("new", "child_only")
        assert parent.get_config("new") is None

    def test_deep_copy_safety(self):
        """Nested structures are deep copied."""
        parent = EmberContext(isolated=True)
        parent.set_config("data", {"list": [1, 2], "dict": {"key": "val"}})

        child = parent.create_child()

        # Modify child's structures
        child.get_config("data")["list"].append(3)
        child.get_config("data")["dict"]["new"] = "value"

        # Parent unaffected
        assert parent.get_config("data")["list"] == [1, 2]
        assert "new" not in parent.get_config("data")["dict"]


class TestThreadSafety:
    """Thread-safe context operations."""

    def test_thread_isolation(self):
        """Each thread has isolated context."""
        results = {}
        barrier = threading.Barrier(2)

        def worker(tid, value):
            with create_context(test=value):
                barrier.wait()  # Synchronize
                results[tid] = get_context().get_config("test")

        threads = [
            threading.Thread(target=worker, args=(1, "thread1")),
            threading.Thread(target=worker, args=(2, "thread2")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == {1: "thread1", 2: "thread2"}

    def test_concurrent_mutations(self):
        """Concurrent config changes are safe."""
        ctx = EmberContext(isolated=True)
        iterations = 100

        def writer(prefix):
            for i in range(iterations):
                ctx.set_config(f"{prefix}.{i}", i)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(writer, f"thread{i}") for i in range(10)]
            for f in futures:
                f.result()

        # Verify all writes succeeded
        for i in range(10):
            for j in range(iterations):
                assert ctx.get_config(f"thread{i}.{j}") == j


class TestAsyncPropagation:
    """Async context propagation via contextvars."""

    @pytest.mark.anyio("asyncio")
    async def test_async_isolation(self):
        """Async tasks maintain context isolation."""
        results = []

        async def task(name, model):
            with create_context(model=model):
                await asyncio.sleep(0.001)
                ctx = get_context()
                results.append((name, ctx.get_config("model")))

        await asyncio.gather(task("t1", "gpt-4"), task("t2", "claude-3"))

        assert sorted(results) == [("t1", "gpt-4"), ("t2", "claude-3")]

    @pytest.mark.anyio("asyncio")
    async def test_context_persistence(self):
        """Context persists across await boundaries."""
        with create_context(test="value"):
            ctx1 = get_context()
            await asyncio.sleep(0.001)
            ctx2 = get_context()

            assert ctx1 is ctx2
            assert ctx2.get_config("test") == "value"


class TestCredentials:
    """Credential management."""

    def test_precedence(self, reset_context):
        """Environment vars do not override stored credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = EmberContext(isolated=True)
            ctx._config_file = Path(tmpdir) / "config.yaml"

            # Create fresh credential manager
            from ember.core.credentials import CredentialManager

            ctx._credential_manager = CredentialManager(ctx._config_file.parent)

            ctx._credentials.save_api_key("openai", "stored-key")

            # Env vars are ignored for credentials
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                assert ctx.get_credential("openai", "OPENAI_API_KEY") == "stored-key"

            # Stored value still works
            with patch.dict(os.environ, {}, clear=True):
                assert ctx.get_credential("openai", "OPENAI_API_KEY") == "stored-key"

    def test_config_overrides_stored_credentials(self, reset_context):
        """Config-provided API keys override legacy credential entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = EmberContext(isolated=True)
            ctx._config_file = Path(tmpdir) / "config.yaml"

            from ember.core.credentials import CredentialManager

            ctx._credential_manager = CredentialManager(ctx._config_file.parent)
            ctx._credentials.save_api_key("openai", "stored-key")
            ctx.set_config("providers.openai.api_key", "config-key")

            with patch.dict(os.environ, {}, clear=True):
                assert ctx.get_credential("openai", "OPENAI_API_KEY") == "config-key"

    def test_crud_operations(self):
        """Create, read, update, delete credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create isolated context with temp directory
            ctx = EmberContext(isolated=True)
            # Override the config directory
            ctx._config_file = Path(tmpdir) / "ember" / "config.yaml"
            ctx._credential_manager = None  # Force re-initialization

            # Create a new credential manager with the temp directory
            from ember.core.credentials import CredentialManager

            ctx._credential_manager = CredentialManager(ctx._config_file.parent)

            # Create
            ctx._credentials.save_api_key("test", "test-api-key-123456")
            assert ctx.get_credential("test", "TEST_API_KEY") == "test-api-key-123456"

            # Update
            ctx._credentials.save_api_key("test", "test-api-key-updated")
            assert ctx.get_credential("test", "TEST_API_KEY") == "test-api-key-updated"

            # List
            assert "test" in ctx._credentials.list_providers()

            # Delete
            # The config file should exist after save_api_key
            assert ctx._credentials.config_file.exists()
            # Now delete should work
            ctx._credentials.delete("test")
            with pytest.raises(CredentialNotFoundError):
                ctx.get_credential("test", "TEST_API_KEY")


class TestPublicAPI:
    """Public API surface."""

    def test_context_manager(self):
        """with_context provides temporary overrides."""
        set_config("test", "original")

        with with_context(test="temporary"):
            assert get_config("test") == "temporary"

        assert get_config("test") == "original"

    def test_shortcuts(self):
        """Config shortcuts work correctly."""
        set_config("test.nested", "value")
        assert get_config("test.nested") == "value"
        assert get_config("missing", "default") == "default"


class TestPersistence:
    """Configuration persistence."""

    def test_save_load_cycle(self):
        """Config survives save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "config.yaml"

            # Save
            ctx1 = EmberContext(isolated=True)
            ctx1._config_file = file_path
            ctx1.set_config("test", {"nested": "value"})
            ctx1.save()

            # Load
            ctx2 = EmberContext(isolated=True)
            ctx2._config_file = file_path
            ctx2.reload()

            assert ctx2.get_config("test.nested") == "value"

    def test_corrupted_file_handling(self):
        """Corrupted configs raise instead of being ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.yaml"
            bad_file.write_text("{ invalid: yaml: }")

            ctx = EmberContext(isolated=True)
            ctx._config_file = bad_file
            with pytest.raises(ValueError):
                ctx.reload()


class TestPerformance:
    """Performance characteristics."""

    def test_config_lookup_speed(self):
        """Nested config lookups are fast."""
        ctx = EmberContext(isolated=True)

        # Build deep config
        for i in range(100):
            ctx.set_config(f"level1.level2.level3.item{i}", i)

        # Measure lookups
        start = time.perf_counter()
        for i in range(1000):
            ctx.get_config(f"level1.level2.level3.item{i % 100}")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.01  # < 10ms for 1k lookups

    def test_child_creation_overhead(self):
        """Child context creation is efficient."""
        parent = EmberContext(isolated=True)

        # Warm up
        parent.create_child()

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            parent.create_child()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1  # < 100ms for 100 children
