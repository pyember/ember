"""Manage configuration and state with Ember contexts.

This example demonstrates:
- Accessing the global context
- Configuration overrides with context managers
- Thread-safe context access
- Nested contexts for isolation

Run with:
    python examples/02_core_concepts/context_management.py
"""

from __future__ import annotations

from ember.context import context, get_config, set_config


def demonstrate_basic_context() -> None:
    """Show basic context access patterns."""
    print("Basic Context Access")
    print("-" * 50)

    # Get the current context
    ctx = context.get()
    print(f"Context type: {type(ctx).__name__}")

    # Read configuration values
    default_model = get_config("models.default", "gpt-4o-mini")
    temperature = get_config("models.temperature", 0.7)
    print(f"Default model: {default_model}")
    print(f"Temperature: {temperature}")
    print()


def demonstrate_context_manager() -> None:
    """Show context manager for temporary overrides."""
    print("Context Manager Overrides")
    print("-" * 50)

    # Get baseline config
    original = get_config("models.default", "gpt-4o-mini")
    print(f"Original default model: {original}")

    # Override within a scope
    with context.manager(models={"default": "gpt-4", "temperature": 0.2}):
        ctx = context.get()
        overridden = ctx.get_config("models.default", "gpt-4o-mini")
        temp = ctx.get_config("models.temperature", 0.7)
        print(f"Inside context manager:")
        print(f"  Model: {overridden}")
        print(f"  Temperature: {temp}")

    # Back to original
    restored = get_config("models.default", "gpt-4o-mini")
    print(f"After context manager: {restored}")
    print()


def demonstrate_nested_contexts() -> None:
    """Show nested context isolation."""
    print("Nested Context Isolation")
    print("-" * 50)

    print("Outer scope:")
    with context.manager(models={"default": "gpt-4"}) as outer:
        outer_model = outer.get_config("models.default")
        print(f"  Model: {outer_model}")

        print("Inner scope (overrides outer):")
        with context.manager(models={"default": "claude-3-5-sonnet"}) as inner:
            inner_model = inner.get_config("models.default")
            print(f"    Model: {inner_model}")

        # Back to outer scope
        restored = outer.get_config("models.default")
        print(f"  Back to outer: {restored}")

    print()


def demonstrate_config_patterns() -> None:
    """Show common configuration patterns."""
    print("Configuration Patterns")
    print("-" * 50)

    # Pattern 1: Development vs Production
    print("1. Environment-based configuration:")
    dev_config = {"models": {"default": "gpt-4o-mini", "temperature": 1.0}}
    prod_config = {"models": {"default": "gpt-4", "temperature": 0.3}}

    for env, config in [("development", dev_config), ("production", prod_config)]:
        with context.manager(**config):
            model = get_config("models.default")
            temp = get_config("models.temperature")
            print(f"   {env}: model={model}, temp={temp}")
    print()

    # Pattern 2: Test isolation
    print("2. Test isolation pattern:")
    with context.manager(test_mode=True, models={"default": "mock-model"}):
        is_test = get_config("test_mode", False)
        model = get_config("models.default")
        print(f"   Test mode: {is_test}, model: {model}")
    print()

    # Pattern 3: Setting config at runtime
    print("3. Runtime configuration:")
    set_config("custom.setting", "my_value")
    value = get_config("custom.setting")
    print(f"   Set custom.setting = {value}")
    print()


def main() -> None:
    """Demonstrate context management."""
    print("Ember Context Management")
    print("=" * 50)
    print()

    demonstrate_basic_context()
    demonstrate_context_manager()
    demonstrate_nested_contexts()
    demonstrate_config_patterns()

    print("Key Takeaways")
    print("-" * 50)
    print("1. context.get() returns the current thread-local context")
    print("2. context.manager() creates scoped configuration overrides")
    print("3. Nested contexts isolate changes automatically")
    print("4. get_config/set_config provide convenient access")
    print()
    print("Next: See error_handling.py for robust patterns")


if __name__ == "__main__":
    main()
