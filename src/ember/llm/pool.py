"""Long-lived provider instances, shared by calls with the same configuration.

A provider owns HTTP connections, so ember builds one per distinct
configuration and reuses it. The key is everything a provider is built from:
provider name, kind, base URL, credential and timeouts. A call made under
`ember.context.override` with a different key therefore gets its own
provider; it never reuses one built for another credential.

Calls lease a provider for the duration of a request. The pool holds at most
`capacity` providers that no call is using; past that, the least recently
used idle provider is closed. A provider in use is never closed by eviction.
"""

import collections
import dataclasses
import threading
from collections.abc import Callable
from typing import TypeAlias

from ember.context import Context, ProviderKind, ProviderSettings, Secret
from ember.llm.provider import Provider

__all__ = ["Lease", "ProviderFactory", "ProviderPool"]

ProviderFactory: TypeAlias = Callable[[ProviderSettings, Context], Provider]
"""Builds a provider from its settings and the active context."""


@dataclasses.dataclass(frozen=True, slots=True)
class _Key:
    name: str
    kind: ProviderKind
    base_url: str | None
    credential: Secret | None
    connect_timeout_s: float
    idle_timeout_s: float


@dataclasses.dataclass(slots=True)
class _Entry:
    provider: Provider
    leases: int = 0


class Lease:
    """One call's use of a pooled provider. Release it exactly once."""

    __slots__ = ("_key", "_pool", "_released", "provider")

    def __init__(self, pool: "ProviderPool", key: _Key, provider: Provider) -> None:
        self._pool = pool
        self._key = key
        self._released = False
        self.provider = provider

    def release(self) -> None:
        """Returns the provider to the pool.

        Raises:
            RuntimeError: The lease was already released.
        """
        if self._released:
            raise RuntimeError("provider lease was already released")
        self._released = True
        self._pool._give_back(self._key)


class ProviderPool:
    """Builds, shares and closes providers. Thread-safe."""

    def __init__(self, factory: ProviderFactory, *, capacity: int = 16) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._factory = factory
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: collections.OrderedDict[_Key, _Entry] = collections.OrderedDict()
        self._closed = False

    def lease(self, settings: ProviderSettings, ctx: Context) -> Lease:
        """Returns a lease on the provider for ``settings`` under ``ctx``.

        A provider is built outside the pool's lock, so a slow build never
        blocks calls to other providers. If two calls build the same provider
        at once, the second one closes its copy and uses the first.

        Raises:
            ConfigError: The factory cannot build the provider.
            RuntimeError: The pool was closed.
        """
        key = _Key(
            name=settings.name,
            kind=settings.kind,
            base_url=settings.base_url,
            credential=settings.credential(),
            connect_timeout_s=ctx.connect_timeout_s,
            idle_timeout_s=ctx.idle_timeout_s,
        )
        lease = self._lease_existing(key)
        if lease is not None:
            return lease
        built = self._factory(settings, ctx)
        with self._lock:
            if self._closed:
                surplus = [built]
                lease = None
            else:
                entry = self._entries.setdefault(key, _Entry(built))
                surplus = [] if entry.provider is built else [built]
                lease = self._take(key, entry)
                surplus += self._evict_idle()
        _close_all(surplus)
        if lease is None:
            raise RuntimeError("the client was closed")
        return lease

    def close(self) -> None:
        """Closes every provider, including any in use, and refuses new leases."""
        with self._lock:
            self._closed = True
            providers = [entry.provider for entry in self._entries.values()]
            self._entries.clear()
        _close_all(providers)

    def size(self) -> int:
        """Returns how many providers the pool holds."""
        with self._lock:
            return len(self._entries)

    def _lease_existing(self, key: _Key) -> Lease | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("the client was closed")
            entry = self._entries.get(key)
            return None if entry is None else self._take(key, entry)

    def _take(self, key: _Key, entry: _Entry) -> Lease:
        entry.leases += 1
        self._entries.move_to_end(key)
        return Lease(self, key, entry.provider)

    def _give_back(self, key: _Key) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases -= 1
            evicted = self._evict_idle()
        _close_all(evicted)

    def _evict_idle(self) -> list[Provider]:
        idle = [key for key, entry in self._entries.items() if entry.leases == 0]
        excess = len(self._entries) - self._capacity
        return [self._entries.pop(key).provider for key in idle[: max(0, excess)]]


def _close_all(providers: list[Provider]) -> None:
    for provider in providers:
        provider.close()
