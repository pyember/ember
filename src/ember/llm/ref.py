"""Model references.

A model is named ``provider/name``. The string is parsed once, at the API
boundary, into a `ModelRef`; nothing downstream splits strings again.
"""

import dataclasses

__all__ = ["ModelRef"]


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRef:
    """A provider and a model name within it.

    Attributes:
        provider: Provider name, for example ``"openai"``.
        name: Model name within the provider. May contain ``:``, ``.``, ``-``,
            ``_`` and further ``/`` characters (for example
            ``"meta-llama/Llama-3.1-8B"`` on an OpenAI-compatible server).
    """

    provider: str
    name: str

    def __post_init__(self) -> None:
        if not self.provider or "/" in self.provider or self.provider != self.provider.strip():
            raise ValueError(f"invalid provider name {self.provider!r}")
        if not self.name or self.name != self.name.strip():
            raise ValueError(f"invalid model name {self.name!r}")

    @classmethod
    def parse(cls, text: str) -> "ModelRef":
        """Parses ``provider/name``; everything after the first ``/`` is the name.

        Raises:
            ValueError: ``text`` has no ``/``, or either side is empty or padded.
        """
        provider, sep, name = text.partition("/")
        if not sep:
            raise ValueError(f"model {text!r} is not of the form provider/name")
        return cls(provider, name)

    def __str__(self) -> str:
        return f"{self.provider}/{self.name}"
