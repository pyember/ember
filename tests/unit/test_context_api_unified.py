"""Test the unified context API implementation.

This test verifies that the new simplified context API maintains all
functionality while providing a cleaner interface.
"""

import pytest

from ember.context import EmberContext, context


class TestUnifiedContextAPI:
    """Test the new unified context API."""

    def test_context_get(self):
        """Test context.get() returns current context."""
        ctx1 = context.get()
        ctx2 = context.get()

        # Should return same instance
        assert ctx1 is ctx2
        assert isinstance(ctx1, EmberContext)

    def test_context_manager_basic(self):
        """Test context.manager() for temporary overrides."""
        # Set a value in current context
        context.get().set_config("test.value", "original")

        # Use context manager to override
        with context.manager(test={"value": "override"}) as ctx:
            assert ctx.get_config("test.value") == "override"
            # Current context should also see override
            assert context.get().get_config("test.value") == "override"

        # After exiting, original value restored
        assert context.get().get_config("test.value") == "original"

    def test_context_manager_nested(self):
        """Test nested context managers."""
        context.get().set_config("models.default", "gpt-3.5-turbo")
        context.get().set_config("models.temperature", 1.0)

        with context.manager(models={"default": "gpt-4"}) as ctx1:
            assert ctx1.get_config("models.default") == "gpt-4"
            assert ctx1.get_config("models.temperature") == 1.0  # Inherited

            with context.manager(models={"temperature": 0.5}) as ctx2:
                assert ctx2.get_config("models.default") == "gpt-4"  # From parent
                assert ctx2.get_config("models.temperature") == 0.5  # Override

        # Original values restored
        assert context.get().get_config("models.default") == "gpt-3.5-turbo"
        assert context.get().get_config("models.temperature") == 1.0

    def test_clean_api_surface(self):
        """Test that only the new API is exposed."""
        import ember.context

        # Should have the new API
        assert hasattr(ember.context, "context")
        assert hasattr(ember.context, "EmberContext")
        assert hasattr(ember.context, "get_config")
        assert hasattr(ember.context, "set_config")

        # Should NOT have old API
        assert not hasattr(ember.context, "get_context")
        assert not hasattr(ember.context, "create_context")
        assert not hasattr(ember.context, "with_context")
        assert not hasattr(ember.context, "current_context")

    def test_config_helpers(self):
        """Test get_config and set_config helpers."""
        from ember.context import get_config, set_config

        # Test set and get
        set_config("helper.test", "value")
        assert get_config("helper.test") == "value"

        # Test default
        assert get_config("helper.missing", "default") == "default"

    def test_thread_safety(self):
        """Test context isolation across threads."""
        import threading

        results = []

        def worker(value):
            with context.manager(worker={"id": value}) as ctx:
                # Each thread should see its own value
                results.append(ctx.get_config("worker.id"))

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have recorded its own value
        assert sorted(results) == [0, 1, 2]

    @pytest.mark.anyio("asyncio")
    async def test_async_context_propagation(self):
        """Test context propagation in async code."""

        async def async_operation():
            # Should see the override from parent
            return context.get().get_config("async.test")

        # Set value in sync context
        context.get().set_config("async.test", "original")

        # Override in async context
        with context.manager(**{"async": {"test": "override"}}):
            result = await async_operation()
            assert result == "override"

        # Original restored
        assert context.get().get_config("async.test") == "original"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
