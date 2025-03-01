import threading
from typing import Dict, Union, Optional
import logging

from ember.core.registry.model.base.schemas.usage import (
    UsageRecord,
    UsageSummary,
    UsageStats,
)


class UsageService:
    """Manages in-memory usage records in a thread-safe manner.

    This class provides functionality to record and retrieve usage statistics
    for individual models using an in-memory store. Thread safety is guaranteed
    by employing a locking mechanism.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initializes a new instance of UsageService.

        Establishes an empty usage summary store and initializes a lock to ensure
        thread-safe operations.

        Args:
            logger (logging.Logger): Logger instance.
        """
        self._lock: threading.Lock = threading.Lock()
        self._usage_summaries: Dict[str, UsageSummary] = {}
        self._logger: logging.Logger = logger or logging.getLogger(
            self.__class__.__name__
        )  # Provide a fallback

    def add_usage_record(
        self, *, model_id: str, usage_stats: Union[UsageStats, dict]
    ) -> None:
        """Records a usage entry for the specified model in a thread-safe manner.

        If the provided `usage_stats` is a dictionary, it is converted into a
        UsageStats instance before recording.

        Args:
            model_id (str): A unique identifier for the model.
            usage_stats (Union[UsageStats, dict]): Usage statistics either as a
                UsageStats instance or as a dict to be converted.

        Returns:
            None.
        """
        with self._lock:
            # Ensure a UsageSummary exists for the model, creating one if necessary.
            usage_summary: UsageSummary = self._usage_summaries.setdefault(
                model_id, UsageSummary(model_name=model_id)
            )
            # Convert dict to a UsageStats instance if needed.
            converted_usage_stats: UsageStats = (
                UsageStats(**usage_stats)
                if isinstance(usage_stats, dict)
                else usage_stats
            )
            usage_record: UsageRecord = UsageRecord(usage_stats=converted_usage_stats)
            usage_summary.add_usage_record(usage_record)

    def get_usage_summary(self, *, model_id: str) -> UsageSummary:
        """Retrieves the current usage summary for the specified model in a thread-safe manner.

        If no summary exists for the given model, a new UsageSummary is created,
        stored, and then returned.

        Args:
            model_id (str): A unique identifier for the model.

        Returns:
            UsageSummary: The current usage summary for the model.
        """
        with self._lock:
            # Use setdefault to ensure the summary is stored for future retrieval.
            return self._usage_summaries.setdefault(
                model_id, UsageSummary(model_name=model_id)
            )
