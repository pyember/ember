PROVIDER_REGISTRY = {}


def register_provider(name: str):
    """Decorator to register a provider class under a given name in the global PROVIDER_REGISTRY."""

    def decorator(cls):
        PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator
