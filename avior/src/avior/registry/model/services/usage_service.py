import threading
from typing import Dict, Union
from src.avior.registry.model.schemas.usage import UsageRecord, UsageSummary, UsageStats


class UsageService:
    """
    Manages usage records for each model, stored in memory.
    Thread-safe with a lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.usage_summaries: Dict[str, UsageSummary] = {}

    def add_usage_record(
        self, model_id: str, usage_stats: Union[UsageStats, dict]
    ) -> None:
        """
        Thread-safe record of usage for a model. Accepts either:
          - usage_stats as an actual UsageStats object
          - usage_stats as a dict (like {"total_tokens": 30, "prompt_tokens": 10, ...})
        """
        with self._lock:
            if model_id not in self.usage_summaries:
                self.usage_summaries[model_id] = UsageSummary(model_name=model_id)

            # Convert dict to a UsageStats if needed
            if isinstance(usage_stats, dict):
                usage_stats = UsageStats(**usage_stats)

            record = UsageRecord(usage_stats=usage_stats)
            self.usage_summaries[model_id].add_record(record)

    def get_usage_summary(self, model_id: str) -> UsageSummary:
        with self._lock:
            return self.usage_summaries.get(model_id, UsageSummary(model_name=model_id))
