"""`Ensemble`: run several operators on one input and keep every outcome.

Members run concurrently through `ember.xcs.map`. A member whose model call
failed for an operational reason (rate limits, overload, server errors,
transport failures, cancellation), or whose own quorum failed, is recorded as
`Failed` and the others carry on. Any other exception, including
authentication, permission, not-found, bad-request and price errors and every
bug in user code, propagates unchanged: those fail every member alike, and a
quorum met despite them would hide an outage.
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Generic, TypeAlias, TypeVar

from ember.errors import ErrorKind, ModelError, QuorumError
from ember.operators.base import In, Operator, Out
from ember.operators.upstream import xcs_map

__all__ = [
    "MEMBER_FAILURE_KINDS",
    "Ensemble",
    "Failed",
    "Outcome",
    "Succeeded",
    "require_quorum",
    "run_members",
]

T = TypeVar("T")
V = TypeVar("V", covariant=True)

#: Model error kinds that fail one member without failing the ensemble.
MEMBER_FAILURE_KINDS = frozenset(
    {
        ErrorKind.RATE_LIMITED,
        ErrorKind.OVERLOADED,
        ErrorKind.SERVER,
        ErrorKind.TRANSPORT,
        ErrorKind.CANCELLED,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Succeeded(Generic[V]):
    """A member that returned a value.

    Attributes:
        member: The member's position in the ensemble.
        value: What it returned.
    """

    member: int
    value: V


@dataclasses.dataclass(frozen=True, slots=True)
class Failed:
    """A member that failed for an operational reason.

    Attributes:
        member: The member's position in the ensemble.
        error: Why it failed.
    """

    member: int
    error: ModelError | QuorumError


Outcome: TypeAlias = Succeeded[T] | Failed


@dataclasses.dataclass(frozen=True, slots=True)
class Ensemble(Operator[In, tuple[Outcome[Out], ...]]):
    """Runs every member on the same input and returns each member's outcome.

    Outcomes are in member order. The call raises `QuorumError`, carrying
    every failure, when fewer than ``quorum`` members succeed.

    Attributes:
        members: The operators to run; at least one.
        quorum: How many members must succeed, from 1 to ``len(members)``.
    """

    members: tuple[Operator[In, Out], ...]
    quorum: int = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple):
            raise TypeError(f"Ensemble members must be a tuple, got {type(self.members)}")
        if not self.members:
            raise ValueError("an Ensemble needs at least one member")
        for index, member in enumerate(self.members):
            if not isinstance(member, Operator):
                raise TypeError(f"Ensemble member {index} is not an Operator; wrap it with @op")
        if not 1 <= self.quorum <= len(self.members):
            raise ValueError(f"quorum must be between 1 and {len(self.members)}, got {self.quorum}")

    def forward(self, input: In) -> tuple[Outcome[Out], ...]:
        outcomes = run_members([functools.partial(member, input) for member in self.members])
        require_quorum(outcomes, self.quorum)
        return outcomes


def run_members(calls: Sequence[Callable[[], T]]) -> tuple[Outcome[T], ...]:
    """Runs ``calls`` concurrently and returns each one's outcome, in order.

    Raises:
        Exception: The first, by position, of the calls' non-operational
            errors, unchanged.
    """
    return tuple(xcs_map(_attempt, enumerate(calls)))


def require_quorum(outcomes: Sequence[Outcome[T]], quorum: int) -> None:
    """Raises `QuorumError` when fewer than ``quorum`` outcomes succeeded."""
    succeeded = sum(1 for outcome in outcomes if isinstance(outcome, Succeeded))
    if succeeded < quorum:
        failures = tuple(
            (outcome.member, outcome.error) for outcome in outcomes if isinstance(outcome, Failed)
        )
        raise QuorumError(required=quorum, succeeded=succeeded, failures=failures)


def _attempt(item: tuple[int, Callable[[], T]]) -> Outcome[T]:
    index, call = item
    try:
        return Succeeded(index, call())
    except ModelError as error:
        if error.kind not in MEMBER_FAILURE_KINDS:
            raise
        return Failed(index, error)
    except QuorumError as error:
        return Failed(index, error)
