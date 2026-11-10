"""`Candidates`: a question and the answers proposed for it."""

import dataclasses

__all__ = ["Candidates"]


@dataclasses.dataclass(frozen=True, slots=True)
class Candidates:
    """The input of `Judge`, `MajorityVote` and `Verifier`.

    Attributes:
        question: What was asked.
        answers: The proposed answers, in member order; at least one.
    """

    question: str
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.answers, tuple):
            raise TypeError(f"Candidates answers must be a tuple, got {type(self.answers)}")
        if not self.answers:
            raise ValueError("Candidates needs at least one answer")
