"""`Verifier`: a model checks each candidate answer against the question.

Every answer is checked by its own call, and the calls run concurrently. The
model is asked to end its reply with a line that reads exactly
``VERDICT: PASS`` or ``VERDICT: FAIL``. The last line that is not blank must
match one of them character for character; anything else, including other
case or spacing, is `Verdict.UNREADABLE`, never guessed.
"""

import dataclasses
import enum
import functools

from ember.llm.result import Result
from ember.operators.base import Operator
from ember.operators.candidates import Candidates
from ember.operators.ensemble import Outcome, run_members
from ember.operators.model_call import ModelCall

__all__ = ["VERIFY_INSTRUCTIONS", "Check", "Verdict", "Verifier"]

VERIFY_INSTRUCTIONS = (
    "Check whether the answer below correctly and completely answers the question. "
    "Explain briefly, then end with a line that reads exactly VERDICT: PASS or VERDICT: FAIL."
)


class Verdict(enum.Enum):
    """What the verifying model concluded about one answer."""

    PASS = "pass"
    FAIL = "fail"
    UNREADABLE = "unreadable"


_VERDICTS = {"VERDICT: PASS": Verdict.PASS, "VERDICT: FAIL": Verdict.FAIL}


@dataclasses.dataclass(frozen=True, slots=True)
class Check:
    """One answer's verification.

    Attributes:
        answer: The answer that was checked.
        verdict: The model's conclusion.
        result: The verifying call's full result.
    """

    answer: str
    verdict: Verdict
    result: Result


@dataclasses.dataclass(frozen=True, slots=True)
class Verifier(Operator[Candidates, tuple[Outcome[Check], ...]]):
    """Checks every candidate answer and returns each check's outcome, in order.

    A check whose call fails for an operational reason is `Failed`, as in
    `Ensemble`; other errors propagate.

    Attributes:
        call: The model call that verifies; it receives one rendered prompt
            per answer.
        instructions: What the verifier is asked to do.
    """

    call: ModelCall
    instructions: str = VERIFY_INSTRUCTIONS

    def forward(self, input: Candidates) -> tuple[Outcome[Check], ...]:
        checks = [functools.partial(self._check, input.question, a) for a in input.answers]
        return run_members(checks)

    def prompt(self, question: str, answer: str) -> str:
        """Returns the prompt that checks ``answer``."""
        return f"{self.instructions}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"

    def _check(self, question: str, answer: str) -> Check:
        result = self.call(self.prompt(question, answer)).result()
        return Check(answer=answer, verdict=read_verdict(result.text), result=result)


def read_verdict(text: str) -> Verdict:
    """Reads the verdict from the last line of ``text`` that is not blank."""
    lines = [line for line in text.splitlines() if line.strip()]
    return _VERDICTS.get(lines[-1], Verdict.UNREADABLE) if lines else Verdict.UNREADABLE
