"""
Dummy Language Model (LM) classes for testing purposes.
"""

class DummyLM:
    """
    A mock LM that returns a static string or can be overridden
    to produce predictable, testable outputs.
    """
    def __call__(self, *, prompt: str, **kwargs) -> str:
        return "DummyLM response"


class FailingDummyLM:
    """
    A mock LM that raises an exception when called, simulating failure.
    """
    def __call__(self, *, prompt: str, **kwargs) -> str:
        raise RuntimeError("Simulated LM failure") 