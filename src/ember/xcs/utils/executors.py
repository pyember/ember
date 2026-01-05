"""Thread pool management helpers for orchestration batching.

The orchestration-aware transformations often need to fan out lightweight tasks
(such as HTTP requests or mocked sleeps) across a small batch. Creating a new
``ThreadPoolExecutor`` for every call adds measurable latency, especially when
many orchestration batches run concurrently under xdist. We maintain a small set
of shared executors instead so that the threads stay warm while still honoring
``Config.max_workers`` and system CPU limits.
"""

from __future__ import annotations

import atexit
import os
import threading
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

# Keep executor construction lightweight and threads warm across calls while
# still respecting isolation between different worker-count configurations.
_CPU_COUNT = os.cpu_count() or 1
_GLOBAL_MAX_WORKERS = max(4, min(32, _CPU_COUNT))


def _build_buckets(limit: int) -> Tuple[int, ...]:
    buckets = []
    size = 1
    while size < limit:
        buckets.append(size)
        size *= 2
    buckets.append(limit)
    return tuple(buckets)


_BUCKETS = _build_buckets(_GLOBAL_MAX_WORKERS)


def _select_bucket(requested: int) -> int:
    if requested <= 0:
        raise ValueError("max_workers must be positive for executor cache")
    target = min(requested, _GLOBAL_MAX_WORKERS)
    index = bisect_left(_BUCKETS, target)
    return _BUCKETS[index]


class _ExecutorCache:
    """Cache thread pools keyed by ``max_workers``.

    The cache never shrinks during the process lifetime; the executors are
    joined during interpreter shutdown via ``atexit``. Access is thread-safe so
    concurrent orchestration batches can share pools safely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executors: Dict[int, ThreadPoolExecutor] = {}
        atexit.register(self._shutdown_all)

    def get(self, max_workers: int) -> ThreadPoolExecutor:
        bucket = _select_bucket(max_workers)
        with self._lock:
            executor = self._executors.get(bucket)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=bucket)
                self._executors[bucket] = executor
                self._warm_executor(executor, bucket)

            return executor

    def _shutdown_all(self) -> None:
        with self._lock:
            executors = dict(self._executors)
            self._executors.clear()
        for executor in executors.values():
            executor.shutdown(wait=True)

    @staticmethod
    def _warm_executor(executor: ThreadPoolExecutor, max_workers: int) -> None:
        # Submit lightweight tasks to spin up the worker threads eagerly so the
        # first real workload doesn't pay the thread creation cost.
        futures = [executor.submit(lambda: None) for _ in range(max_workers)]
        for future in futures:
            future.result()


_EXECUTOR_CACHE = _ExecutorCache()

# Prime a small executor at import time to keep orchestration batching snappy.
_EXECUTOR_CACHE.get(min(4, _GLOBAL_MAX_WORKERS))


def get_shared_executor(max_workers: int) -> ThreadPoolExecutor:
    """Return a shared thread pool sized for ``max_workers``.

    Args:
        max_workers: Target upper bound on worker threads. The value must be a
            positive integer. Requests are clamped to the machine CPU count and
            bucketed, so the returned executor might allow slightly more
            concurrency than requested but never more than the global limit.

    Returns:
        ThreadPoolExecutor: A reusable executor instance.
    """

    return _EXECUTOR_CACHE.get(max_workers)


__all__ = ["get_shared_executor"]
