"""`Judge`: one model reads the question and every candidate and answers."""

import dataclasses

from ember.llm.result import Result
from ember.operators.base import Operator
from ember.operators.candidates import Candidates
from ember.operators.model_call import ModelCall

__all__ = ["JUDGE_INSTRUCTIONS", "Judge"]

JUDGE_INSTRUCTIONS = (
    "Several answers to the question below were written independently. Read all of them, "
    "work out what is correct, and write the single best answer to the question. "
    "Reply with that answer only."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Judge(Operator[Candidates, Result]):
    """Synthesizes one answer from the question and all of its candidates.

    Attributes:
        call: The model call that judges; it receives the rendered prompt.
        instructions: What the judge is asked to do, placed before the
            question and the numbered answers.
    """

    call: ModelCall
    instructions: str = JUDGE_INSTRUCTIONS

    def forward(self, input: Candidates) -> Result:
        return self.call(self.prompt(input))

    def prompt(self, candidates: Candidates) -> str:
        """Returns the prompt the judge receives for ``candidates``."""
        answers = "\n\n".join(
            f"Answer {number}:\n{answer}"
            for number, answer in enumerate(candidates.answers, start=1)
        )
        return f"{self.instructions}\n\nQuestion:\n{candidates.question}\n\n{answers}"
