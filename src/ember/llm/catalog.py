"""The model catalog: one table of models, loaded once.

Every model ember knows by name is a `ModelSpec` read from ``catalog.toml``,
which ships inside the package. Nothing else in ember lists models.

Resolution, done once per call at the API boundary:

- A catalog id (``"openai/gpt-5.5"``), ``provider/wire_id`` (a model's pinned
  snapshot id) or bare alias (``"gpt-5.5"``) resolves to its spec.
- ``provider/name`` with a configured provider but no catalog entry resolves to
  a spec marked unknown: ``name`` is sent as the wire id, and nothing is known
  about the model's limits, settings or price.
- A bare name that is not an alias, or a provider that is not configured,
  raises ``ModelError(NOT_FOUND)``.

There is no prefix matching, no live discovery and no network access.

Example:
    >>> from ember.llm import catalog
    >>> catalog.default().resolve("gpt-5.5").ref
    ModelRef(provider='openai', name='gpt-5.5')
"""

import datetime
import decimal
import difflib
import importlib.resources
import os
import threading
import tomllib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast, get_args

from ember.context import ProviderKind, ProviderSettings
from ember.errors import ErrorKind, ModelError
from ember.llm.ref import ModelRef
from ember.llm.spec import (
    SAMPLING_FIELDS,
    LongContextRates,
    ModelSpec,
    Rates,
    SamplingField,
    WireApi,
)

__all__ = ["Catalog", "CatalogError", "default", "reset"]

# The wire API for a model the catalog does not list, by provider kind. OpenAI
# recommends its Responses API "for all new projects"; an OpenAI-compatible
# server speaks Chat Completions.
_UNKNOWN_MODEL_API: Mapping[ProviderKind, WireApi] = MappingProxyType(
    {
        "openai": "openai_responses",
        "anthropic": "anthropic_messages",
        "gemini": "gemini",
        "openai_compatible": "openai_chat",
    }
)
_WIRE_APIS: frozenset[str] = frozenset(get_args(WireApi))
_NO_PROVIDERS: Mapping[str, ProviderSettings] = MappingProxyType({})


class CatalogError(ValueError):
    """``catalog.toml`` is malformed: an unknown key, a missing field or a clash."""


class Catalog:
    """An immutable set of model specs with an O(1) name index."""

    __slots__ = ("_by_name", "_specs")

    def __init__(self, specs: tuple[ModelSpec, ...]) -> None:
        """Indexes ``specs`` by id, ``provider/wire_id`` and alias.

        Raises:
            CatalogError: Two specs claim the same name.
        """
        self._specs = tuple(sorted(specs, key=lambda spec: str(spec.ref)))
        self._by_name: Mapping[str, ModelSpec] = MappingProxyType(_index(self._specs))

    @classmethod
    def from_toml(cls, text: str) -> "Catalog":
        """Parses a catalog file.

        Raises:
            CatalogError: The text is not valid TOML or breaks the catalog schema.
        """
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise CatalogError(f"catalog is not valid TOML: {exc}") from exc
        reader = _Table(data, "catalog")
        entries = reader.tables("model")
        reader.finish()
        return cls(tuple(_spec(entry) for entry in entries))

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> ModelSpec | None:
        """Returns the spec ``name`` names exactly, or ``None``."""
        return self._by_name.get(name)

    def resolve(
        self,
        model: str | ModelRef,
        *,
        providers: Mapping[str, ProviderSettings] = _NO_PROVIDERS,
    ) -> ModelSpec:
        """Resolves a model name to its spec, as the module docstring describes.

        Args:
            model: ``provider/name``, a bare alias, or a parsed `ModelRef`.
            providers: The configured providers, which decide whether an
                uncatalogued ``provider/name`` can be sent and over which API.
                With none, only catalog models resolve.

        Raises:
            ModelError: ``NOT_FOUND`` for a bare name that is not an alias or a
                provider that is not configured.
            ValueError: ``model`` is not a well-formed ``provider/name``.
        """
        spec = self._by_name.get(str(model))
        if spec is not None:
            return spec
        if isinstance(model, str) and "/" not in model:
            raise ModelError(
                ErrorKind.NOT_FOUND,
                provider="",
                model=model,
                reason=f"not a catalog model or alias{self._suggestion(model)}; "
                "name other models as provider/name",
            )
        ref = model if isinstance(model, ModelRef) else ModelRef.parse(model)
        settings = providers.get(ref.provider)
        if settings is None:
            configured = ", ".join(sorted(providers)) or "none"
            raise ModelError(
                ErrorKind.NOT_FOUND,
                provider=ref.provider,
                model=ref.name,
                reason=f"unknown provider; configured providers: {configured}",
            )
        return ModelSpec(
            ref=ref,
            wire_id=ref.name,
            api=_UNKNOWN_MODEL_API[settings.kind],
            accepts=frozenset(),
            known=False,
        )

    def _suggestion(self, name: str) -> str:
        close = difflib.get_close_matches(name, self._by_name, n=3)
        return f" (did you mean {', '.join(close)}?)" if close else ""


def _index(specs: tuple[ModelSpec, ...]) -> dict[str, ModelSpec]:
    index: dict[str, ModelSpec] = {}
    for spec in specs:
        names = {str(spec.ref), f"{spec.ref.provider}/{spec.wire_id}", *spec.aliases}
        for name in names:
            claimed = index.setdefault(name, spec)
            if claimed is not spec:
                raise CatalogError(f"{name!r} names both {claimed.ref} and {spec.ref}")
    return index


def _spec(entry: Mapping[str, object]) -> ModelSpec:
    table = _Table(entry, f"model {entry.get('id')!r}")
    ref = ModelRef.parse(table.text("id"))
    api = table.text("api")
    if api not in _WIRE_APIS:
        raise CatalogError(f"{ref}: api must be one of {sorted(_WIRE_APIS)}, got {api!r}")
    aliases = tuple(table.texts("aliases"))
    if any("/" in alias for alias in aliases):
        raise CatalogError(f"{ref}: aliases are bare names without '/'")
    rates = table.optional_table("rates")
    spec = ModelSpec(
        ref=ref,
        wire_id=table.text("wire_id"),
        api=cast(WireApi, api),
        aliases=aliases,
        context_window=table.optional_count("context_window"),
        max_input_tokens=table.optional_count("max_input_tokens"),
        max_output_tokens=table.optional_count("max_output_tokens"),
        accepts=_sampling_fields(ref, table.texts("accepts")),
        reasoning_efforts=frozenset(table.texts("reasoning_efforts")),
        rates=None if rates is None else _rates(rates, f"{ref} rates"),
        sources=tuple(table.texts("sources")),
        as_of=table.date("as_of"),
    )
    table.finish()
    return spec


def _sampling_fields(ref: ModelRef, names: list[str]) -> frozenset[SamplingField]:
    fields = frozenset(field for field in SAMPLING_FIELDS if field in names)
    unknown = set(names) - fields
    if unknown:
        raise CatalogError(f"{ref}: accepts names unknown settings {sorted(unknown)}")
    return fields


def _rates(raw: Mapping[str, object], where: str) -> Rates:
    table = _Table(raw, where)
    source_url = table.text("source_url")
    as_of = table.date("as_of")
    long = table.optional_table("long_context")
    rates = _rate_values(table, source_url, as_of)
    table.finish()
    if long is None:
        return rates
    long_table = _Table(long, f"{where} long_context")
    threshold = long_table.count("above_prompt_tokens")
    long_rates = _rate_values(long_table, source_url, as_of)
    long_table.finish()
    return Rates(
        input=rates.input,
        cache_read=rates.cache_read,
        cache_write=rates.cache_write,
        output=rates.output,
        source_url=source_url,
        as_of=as_of,
        long_context=LongContextRates(above_prompt_tokens=threshold, rates=long_rates),
    )


def _rate_values(table: "_Table", source_url: str, as_of: datetime.date) -> Rates:
    return Rates(
        input=table.dollars("input"),
        cache_read=table.optional_dollars("cache_read"),
        cache_write=table.optional_dollars("cache_write"),
        output=table.dollars("output"),
        source_url=source_url,
        as_of=as_of,
    )


class _Table:
    """Reads typed fields from one TOML table and rejects keys nobody read."""

    def __init__(self, data: Mapping[str, object], where: str) -> None:
        self._data = data
        self._where = where
        self._read: set[str] = set()

    def _get(self, key: str) -> object:
        self._read.add(key)
        if key not in self._data:
            raise CatalogError(f"{self._where}: missing {key!r}")
        return self._data[key]

    def text(self, key: str) -> str:
        value = self._get(key)
        if not isinstance(value, str) or not value:
            raise CatalogError(f"{self._where}: {key} must be a non-empty string")
        return value

    def texts(self, key: str) -> list[str]:
        value = self._get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise CatalogError(f"{self._where}: {key} must be a list of non-empty strings")
        if len(set(value)) != len(value):
            raise CatalogError(f"{self._where}: {key} has duplicates")
        return value

    def count(self, key: str) -> int:
        value = self._get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CatalogError(f"{self._where}: {key} must be a positive integer")
        return value

    def optional_count(self, key: str) -> int | None:
        return self.count(key) if self._present(key) else None

    def date(self, key: str) -> datetime.date:
        value = self._get(key)
        if not isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
            raise CatalogError(f"{self._where}: {key} must be a date like 2026-11-07")
        return value

    def dollars(self, key: str) -> decimal.Decimal:
        value = self._get(key)
        if not isinstance(value, str):
            raise CatalogError(f'{self._where}: {key} must be a decimal string like "1.25"')
        try:
            amount = decimal.Decimal(value)
        except decimal.InvalidOperation:
            raise CatalogError(f"{self._where}: {key} is not a decimal: {value!r}") from None
        if not amount.is_finite() or amount < 0:
            raise CatalogError(f"{self._where}: {key} must be a non-negative amount")
        return amount

    def optional_dollars(self, key: str) -> decimal.Decimal | None:
        return self.dollars(key) if self._present(key) else None

    def tables(self, key: str) -> list[Mapping[str, object]]:
        value = self._get(key)
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise CatalogError(f"{self._where}: {key} must be an array of tables")
        return value

    def optional_table(self, key: str) -> Mapping[str, object] | None:
        if not self._present(key):
            return None
        value = self._get(key)
        if not isinstance(value, dict):
            raise CatalogError(f"{self._where}: {key} must be a table")
        return cast(Mapping[str, object], value)

    def finish(self) -> None:
        unknown = set(self._data) - self._read
        if unknown:
            raise CatalogError(f"{self._where}: unknown keys {sorted(unknown)}")

    def _present(self, key: str) -> bool:
        self._read.add(key)
        return key in self._data


# The packaged catalog, loaded on first use. Owner: this module. Lifetime: the
# process; `reset()` drops it and a forked child gets a fresh lock.
_lock = threading.Lock()
_default: Catalog | None = None


def default() -> Catalog:
    """Returns the packaged catalog, reading ``catalog.toml`` on the first call only."""
    global _default
    catalog = _default
    if catalog is not None:
        return catalog
    with _lock:
        if _default is None:
            text = importlib.resources.files("ember.llm").joinpath("catalog.toml").read_text()
            _default = Catalog.from_toml(text)
        return _default


def reset() -> None:
    """Forgets the loaded catalog, so the next `default()` reads the file again."""
    global _default
    with _lock:
        _default = None


def _renew_lock_in_child() -> None:
    global _lock
    _lock = threading.Lock()


os.register_at_fork(after_in_child=_renew_lock_in_child)
