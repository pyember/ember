"""Track divergences between estimated and provider-reported costs.

The tracker helps validate pricing tables by comparing Ember's estimates to
actual charges reported by providers.

Examples:
    >>> from ember.models.pricing.tracker import CostTracker
    >>> tracker = CostTracker()
    >>> tracker.record_usage(UsageStats(cost_usd=0.1, actual_cost_usd=0.11), "demo")

"""

import logging

from ember.models.schemas import UsageStats

logger = logging.getLogger(__name__)


class CostTracker:
    """Aggregate accuracy metrics for estimated versus actual costs.

    Attributes:
        _total_estimated: Running sum of estimated costs.
        _total_actual: Running sum of actual costs.
        _reconciliation_count: Number of records with actual cost data.
        _max_deviation: Largest absolute deviation observed in USD.

    Examples:
        >>> tracker = CostTracker()
        >>> tracker.record_usage(UsageStats(cost_usd=1.0, actual_cost_usd=0.9), "demo")
    """

    def __init__(self) -> None:
        self._total_estimated = 0.0
        self._total_actual = 0.0
        self._reconciliation_count = 0
        self._max_deviation = 0.0

    def record_usage(self, usage: UsageStats, model_id: str) -> None:
        """Record a usage sample and update reconciliation metrics.

        Args:
            usage: Usage metrics including estimated and optional actual cost.
            model_id: Identifier used when logging discrepancies.

        Returns:
            None

        Examples:
            >>> tracker = CostTracker()
            >>> tracker.record_usage(UsageStats(cost_usd=0.2), 'demo')
        """
        if usage.cost_usd is None:
            return

        self._total_estimated += usage.cost_usd

        # If we have actual cost, compare
        if usage.actual_cost_usd is not None:
            self._total_actual += usage.actual_cost_usd
            self._reconciliation_count += 1

            # Calculate deviation
            deviation = abs(usage.cost_usd - usage.actual_cost_usd)
            deviation_pct = (
                (deviation / usage.actual_cost_usd * 100) if usage.actual_cost_usd > 0 else 0
            )

            # Track max deviation
            if deviation > self._max_deviation:
                self._max_deviation = deviation

            # Log significant discrepancies (>5%) at info level for opt-in visibility
            if deviation_pct > 5:
                logger.info(
                    f"Cost discrepancy for {model_id}: "
                    f"estimated=${usage.cost_usd:.6f}, "
                    f"actual=${usage.actual_cost_usd:.6f} "
                    f"({deviation_pct:.1f}% difference)"
                )
            elif deviation_pct > 0:
                logger.debug(f"Cost reconciliation for {model_id}: deviation={deviation_pct:.1f}%")

    def get_accuracy_metrics(self) -> dict[str, float | int]:
        """Summarize reconciliation performance across all samples.

        Returns:
            dict[str, float | int]: Metrics including counts, accuracy
            percentage, and deviation values.

        Examples:
            >>> CostTracker().get_accuracy_metrics()['reconciliation_count']
            0
        """
        if self._reconciliation_count == 0:
            return {"reconciliation_count": 0, "accuracy_pct": 0.0, "max_deviation_usd": 0.0}

        accuracy = 100.0
        if self._total_actual > 0:
            accuracy = (
                100.0 - abs(self._total_estimated - self._total_actual) / self._total_actual * 100
            )
        # Clamp to [0, 100] for nicer reporting
        if accuracy < 0.0:
            accuracy = 0.0
        elif accuracy > 100.0:
            accuracy = 100.0

        return {
            "reconciliation_count": self._reconciliation_count,
            "accuracy_pct": round(accuracy, 2),
            "max_deviation_usd": self._max_deviation,
            "total_estimated_usd": self._total_estimated,
            "total_actual_usd": self._total_actual,
        }


# Global tracker instance
_cost_tracker = CostTracker()


def track_usage(usage: UsageStats, model_id: str) -> None:
    """Record usage with the module-level tracker.

    Args:
        usage: Usage metrics including estimated and optional actual cost.
        model_id: Identifier used when logging discrepancies.

    Returns:
        None

    Examples:
        >>> track_usage(UsageStats(cost_usd=0.1), 'demo')
    """
    _cost_tracker.record_usage(usage, model_id)


def get_cost_accuracy() -> dict[str, float | int]:
    """Retrieve accuracy metrics from the shared tracker.

    Returns:
        dict[str, float | int]: Same shape as ``CostTracker.get_accuracy_metrics``.

    Examples:
        >>> get_cost_accuracy()['reconciliation_count']
        0
    """
    return _cost_tracker.get_accuracy_metrics()
