"""Utility helpers for merging discovery data with bootstrap catalogs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping

from ember.models.discovery.types import DiscoveredModel, ModelKey, OverrideSpec

ModelInfoLike = Mapping[str, Any]
MergedCatalog = Dict[str, Dict[str, Any]]


def merge_catalog(
    *,
    dynamic: Mapping[str, DiscoveredModel],
    bootstrap: Mapping[str, ModelInfoLike],
    overrides: Mapping[str, OverrideSpec],
) -> MergedCatalog:
    """Merge dynamic discovery, bootstrap metadata, and user overrides."""

    merged: MergedCatalog = {}

    # Seed with bootstrap data so curated fields win by default.
    for key, info in bootstrap.items():
        record = _normalize_bootstrap_entry(key, info)
        merged_key = record["_model_key"]
        merged[merged_key] = record

    # Layer dynamic discovery.
    for key, model in dynamic.items():
        model_key = key if ":" in key else model.model_key()
        record = merged.setdefault(
            model_key,
            {
                "provider": model.provider,
                "id": model.id,
                "aliases": [model.id],
            },
        )
        _apply_dynamic(record, model)

    # Apply overrides last.
    for key, override in overrides.items():
        model_key = key if ":" in key else _infer_override_key(key, merged)
        if model_key not in merged:
            # Create a stub entry so overrides can introduce new models explicitly.
            provider, vendor_id = ModelKey.split(model_key)
            merged[model_key] = {
                "provider": provider,
                "id": vendor_id,
                "aliases": [vendor_id],
            }
        _apply_override(merged[model_key], override)

    # Finalize aliases/capabilities as sorted, deduplicated tuples for stability.
    for record in merged.values():
        _dedupe_sequence(record, "aliases")
        _dedupe_sequence(record, "capabilities")
        record.pop("_model_key", None)

    return merged


def _normalize_bootstrap_entry(key: str, info: ModelInfoLike) -> Dict[str, Any]:
    data = _to_mutable(info)
    provider = data.get("provider")
    if not provider:
        raise ValueError(
            f"Bootstrap entry '{key}' missing provider field; cannot normalize catalog key"
        )
    model_id = data.get("id", key)
    model_key = ModelKey.to_key(provider, model_id)
    data.setdefault("id", model_id)
    aliases = data.get("aliases")
    if not aliases:
        data["aliases"] = [model_id]
    else:
        data["aliases"] = list(aliases)
    capabilities = data.get("capabilities")
    if capabilities:
        data["capabilities"] = list(capabilities)
    data["_model_key"] = model_key
    return data


def _to_mutable(value: Any) -> Dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported bootstrap entry type: {type(value)!r}")


def _apply_dynamic(record: MutableMapping[str, Any], model: DiscoveredModel) -> None:
    record.setdefault("provider", model.provider)
    record.setdefault("id", model.id)
    aliases = record.setdefault("aliases", [])
    if not isinstance(aliases, list):
        aliases = list(aliases)
        record["aliases"] = aliases
    aliases.append(model.id)
    if model.display_name and model.display_name not in aliases:
        aliases.append(model.display_name)
    if model.description:
        record.setdefault("description", model.description)
    if model.context_window_in is not None:
        record.setdefault("context_window", model.context_window_in)
    if model.context_window_out is not None:
        record.setdefault("context_window_out", model.context_window_out)
    if model.capabilities:
        existing = record.setdefault("capabilities", [])
        if not isinstance(existing, list):
            existing = list(existing)
            record["capabilities"] = existing
        for capability in model.capabilities:
            if capability not in existing:
                existing.append(capability)
    if model.region_scope:
        regions = record.setdefault("region_scope", [])
        if not isinstance(regions, list):
            regions = list(regions)
            record["region_scope"] = regions
        for region in model.region_scope:
            if region not in regions:
                regions.append(region)
    if model.raw_payload:
        record.setdefault("discovery_payload", dict(model.raw_payload))


def _apply_override(record: MutableMapping[str, Any], override: OverrideSpec) -> None:
    if "description" in override:
        record["description"] = override["description"]
    if "pricing" in override:
        record["pricing_override"] = override["pricing"]
    if "hidden" in override:
        record["hidden"] = override["hidden"]
    if "aliases" in override:
        record["aliases"] = list(override["aliases"])
    if "capabilities" in override:
        record["capabilities"] = list(override["capabilities"])
    if "context_window" in override:
        record["context_window"] = override["context_window"]
    if "context_window_out" in override:
        record["context_window_out"] = override["context_window_out"]


def _infer_override_key(key: str, merged: MergedCatalog) -> str:
    if ":" in key:
        return key
    # Attempt to match legacy keys by model id.
    for model_key, record in merged.items():
        if key in record.get("aliases", []):
            return model_key
    raise KeyError(f"Override '{key}' does not match any known model. Use provider-qualified keys.")


def _dedupe_sequence(record: MutableMapping[str, Any], field: str) -> None:
    if field not in record:
        return
    values = record[field]
    if not values:
        record[field] = []
        return
    deduped = sorted({v for v in values if v})
    record[field] = deduped


__all__ = ["merge_catalog", "ModelInfoLike", "MergedCatalog"]
