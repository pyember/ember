"""
Dummy LM module implementations for operator wrapper tests.

These dummy modules simulate LM behavior by returning fixed responses or throwing exceptions.
"""


class DummyLM:
    def __init__(self, response: str = "dummy response"):
        self.response = response

    def __call__(self, *, prompt: str) -> str:
        return f"{self.response} for: {prompt}"


class FailingDummyLM:
    def __call__(self, *, prompt: str) -> str:
        raise Exception("LM failure")
