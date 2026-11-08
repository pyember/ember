"""`MajorityVote`: the answer most candidates agree on.

Answers are compared after a normalization the caller chooses, and a tie is
settled by a policy the caller chooses; neither has a default.

Example:
    >>> from ember.operators import Candidates, MajorityVote, TiePolicy, fold_answer
    >>> vote = MajorityVote(normalize=fold_answer, tie=TiePolicy.FIRST_SEEN)
    >>> vote(Candidates("capital of France?", ("Paris", "paris ", "Lyon"))).answer
    'Paris'
"""

import dataclasses
import enum
from collections.abc import Callable

from ember.errors import TieError
from ember.operators.base import Operator
from ember.operators.candidates import Candidates

__all__ = ["MajorityVote", "TiePolicy", "Vote", "fold_answer"]


class TiePolicy(enum.Enum):
    """What `MajorityVote` does when several answers share the most votes."""

    FIRST_SEEN = "first_seen"
    """The tied answer that appears first among the candidates wins."""

    RAISE = "raise"
    """Raise `TieError`."""


def fold_answer(text: str) -> str:
    """Collapses runs of whitespace to one space, strips the ends, and casefolds."""
    return " ".join(text.split()).casefold()


@dataclasses.dataclass(frozen=True, slots=True)
class Vote:
    """The outcome of a vote.

    Attributes:
        answer: The winning answer as first written, before normalization.
        votes: Votes the winning answer received.
        tally: Each normalized answer with its votes, in the order first seen.
    """

    answer: str
    votes: int
    tally: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class MajorityVote(Operator[Candidates, Vote]):
    """Picks the answer with the most votes (a plurality).

    Attributes:
        normalize: Maps an answer to the key answers are compared by.
        tie: How a tie for the most votes is settled.
    """

    normalize: Callable[[str], str]
    tie: TiePolicy

    def forward(self, input: Candidates) -> Vote:
        votes: dict[str, int] = {}
        first_written: dict[str, str] = {}
        for answer in input.answers:
            key = self.normalize(answer)
            votes[key] = votes.get(key, 0) + 1
            first_written.setdefault(key, answer)
        most = max(votes.values())
        leaders = tuple(key for key, count in votes.items() if count == most)
        if len(leaders) > 1 and self.tie is TiePolicy.RAISE:
            raise TieError(leaders, most)
        return Vote(answer=first_written[leaders[0]], votes=most, tally=tuple(votes.items()))
