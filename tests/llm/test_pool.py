"""The provider pool: keyed, bounded, built outside its lock."""

import threading
from collections.abc import Callable, Iterator

import pytest

from ember.context import Context, ProviderSettings, Secret
from ember.llm.pool import ProviderPool
from ember.llm.provider import Decoded, StreamItem
from ember.llm.request import Request
from ember.llm.spec import ModelSpec

CTX = Context(providers={})


class Built:
    """A distinct provider object per build, counting its closes."""

    def __init__(self, name: str) -> None:
        self.label = name
        self.closed = 0

    @property
    def name(self) -> str:
        return self.label

    def send(self, spec: ModelSpec, request: Request) -> Decoded:
        raise NotImplementedError

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed += 1


def _settings(name: str, key: str | None = None) -> ProviderSettings:
    return ProviderSettings(
        name=name,
        kind="openai_compatible",
        api_key=None if key is None else Secret(key),
        base_url="http://h/v1",
    )


def _factory(log: list[Built]) -> Callable[[ProviderSettings, Context], Built]:
    def build(settings: ProviderSettings, ctx: Context) -> Built:
        built = Built(settings.name)
        log.append(built)
        return built

    return build


def test_same_settings_share_one_provider_and_credentials_split_them() -> None:
    log: list[Built] = []
    pool = ProviderPool(_factory(log))
    a = pool.lease(_settings("p", "key-aaaa"), CTX)
    b = pool.lease(_settings("p", "key-aaaa"), CTX)
    c = pool.lease(_settings("p", "key-bbbb"), CTX)
    assert a.provider is b.provider is log[0]
    assert c.provider is log[1] and len(log) == 2


def test_idle_providers_past_capacity_are_closed_oldest_first() -> None:
    log: list[Built] = []
    pool = ProviderPool(_factory(log), capacity=2)
    for name in ("a", "b", "c"):
        pool.lease(_settings(name), CTX).release()
    assert [built.closed for built in log] == [1, 0, 0]
    assert pool.size() == 2


def test_a_provider_in_use_is_never_evicted() -> None:
    log: list[Built] = []
    pool = ProviderPool(_factory(log), capacity=1)
    held = pool.lease(_settings("a"), CTX)
    pool.lease(_settings("b"), CTX).release()
    assert [built.closed for built in log] == [0, 1]
    held.release()
    assert pool.size() == 1


def test_a_lease_releases_exactly_once() -> None:
    pool = ProviderPool(_factory([]))
    lease = pool.lease(_settings("a"), CTX)
    lease.release()
    with pytest.raises(RuntimeError, match="already released"):
        lease.release()


def test_a_slow_build_does_not_block_other_providers() -> None:
    entered, finish = threading.Event(), threading.Event()
    log: list[Built] = []

    def build(settings: ProviderSettings, ctx: Context) -> Built:
        if settings.name == "slow":
            entered.set()
            assert finish.wait(timeout=10)
        built = Built(settings.name)
        log.append(built)
        return built

    pool = ProviderPool(build)
    slow = threading.Thread(target=lambda: pool.lease(_settings("slow"), CTX))
    slow.start()
    assert entered.wait(timeout=10)
    fast = pool.lease(_settings("fast"), CTX)
    assert fast.provider.name == "fast"
    finish.set()
    slow.join(timeout=10)
    assert [built.label for built in log] == ["fast", "slow"]


def test_concurrent_builds_of_one_provider_keep_one_and_close_the_other() -> None:
    barrier = threading.Barrier(2)
    log: list[Built] = []

    def build(settings: ProviderSettings, ctx: Context) -> Built:
        barrier.wait(timeout=10)
        built = Built(settings.name)
        log.append(built)
        return built

    pool = ProviderPool(build)
    leases = []
    threads = [
        threading.Thread(target=lambda: leases.append(pool.lease(_settings("p"), CTX)))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert leases[0].provider is leases[1].provider
    assert sorted(built.closed for built in log) == [0, 1]
    assert pool.size() == 1


def test_a_closed_pool_closes_everything_and_refuses_leases() -> None:
    log: list[Built] = []
    pool = ProviderPool(_factory(log))
    pool.lease(_settings("a"), CTX)
    pool.close()
    assert [built.closed for built in log] == [1]
    with pytest.raises(RuntimeError, match="closed"):
        pool.lease(_settings("a"), CTX)
