"""Configuration objects for XCS public APIs.

The configuration surface intentionally stays compact so that most users rely on
defaults while advanced callers can make explicit trade-offs. All parameters are
optional and validated eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ember.xcs.errors import XCSError


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime configuration for XCS transformations.

    Attributes:
      parallel: Enables parallel execution when `True`.
      cache: Enables result caching for deterministic operators when `True`.
      profile: Forces profiler collection for decorated functions.
      max_workers: Upper bound on worker threads. `None` keeps the runtime
        default.
      max_memory_mb: Soft memory ceiling in megabytes. `None` disables the
        guard.
    """

    parallel: bool = True
    cache: bool = True
    profile: bool = False
    max_workers: Optional[int] = None
    max_memory_mb: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration values eagerly."""
        if self.max_workers is not None and self.max_workers <= 0:
            raise XCSError("max_workers must be a positive integer when provided")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise XCSError("max_memory_mb must be a positive integer when provided")

    def to_dict(self) -> Mapping[str, Any]:
        """Return a read-only mapping representation of the config."""
        return {
            "parallel": self.parallel,
            "cache": self.cache,
            "profile": self.profile,
            "max_workers": self.max_workers,
            "max_memory_mb": self.max_memory_mb,
        }

    def apply_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "Config":
        """Return a new config with `overrides` applied.

        Args:
          overrides: Mapping of configuration keys to replacement values.

        Returns:
          Config: A new instance containing the merged settings.

        Raises:
          XCSError: If unknown keys are supplied.
        """
        if not overrides:
            return self

        invalid = set(overrides).difference(self.to_dict())
        if invalid:
            raise XCSError(f"Unknown config keys: {sorted(invalid)}")

        payload: Dict[str, Any] = dict(self.to_dict())
        payload.update(overrides)
        return Config(
            parallel=payload["parallel"],
            cache=payload["cache"],
            profile=payload["profile"],
            max_workers=payload["max_workers"],
            max_memory_mb=payload["max_memory_mb"],
        )


class Presets:
    """Curated configuration presets for common scenarios."""

    SECURE = Config(cache=False, profile=False)
    DEBUG = Config(profile=True)
    LIGHTWEIGHT = Config(parallel=True, cache=True, max_workers=2)
    SERIAL = Config(parallel=False, cache=True)


__all__ = ["Config", "Presets"]
