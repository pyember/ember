from __future__ import annotations

import re
import abc
from typing import Any, Optional, TypeVar, Generic

# We no longer have ExtractionResult here.
T_out = TypeVar("T_out")
T_truth = TypeVar("T_truth")


class IOutputExtractor(Generic[T_out, T_truth], metaclass=abc.ABCMeta):
    """
    Interface for extracting or converting raw system output into a refined form.

    Implementers should provide an `extract` method.
    """

    @abc.abstractmethod
    def extract(self, system_output: T_out, **kwargs: Any) -> T_truth:
        """Extract or convert the system output into a different form.

        Args:
            system_output (T_out): The raw output from the system.
            **kwargs: Additional keyword arguments.

        Returns:
            T_truth: The extracted or converted output.
        """
        raise NotImplementedError


class RegexExtractor(IOutputExtractor[str, str]):
    """
    Extractor that uses a regular expression to select a substring.

    Returns the first captured group, or an empty string if no match is found.
    """

    def __init__(self, pattern: str) -> None:
        self.compiled_pattern: re.Pattern = re.compile(pattern)

    def extract(self, system_output: str, **kwargs: Any) -> str:
        """Extracts a substring from the system output using the specified regular expression.

        Args:
            system_output (str): The raw output from the system.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            str: The first captured group from the match, or an empty string if no match is found.
        """
        match: Optional[re.Match] = self.compiled_pattern.search(system_output)
        if match is None:
            return ""
        return match.group(1)
