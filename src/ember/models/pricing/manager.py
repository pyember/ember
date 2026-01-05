"""Pricing helpers backed by centralized configuration.

The pricing system loads canonical rates from ``pricing.yaml`` and merges
overrides from ``~/.ember/config.yaml`` via the ``models.overrides`` section.

Examples:
    >>> from ember.models.pricing.manager import get_model_cost
    >>> cost = get_model_cost("gpt-4")
    >>> sorted(cost.keys())
    ['context', 'input', 'output']
"""

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, TypedDict

import yaml

from ember.models.catalog import get_model_overrides
from ember.models.discovery.types import ModelKey

logger = logging.getLogger(__name__)


class ModelPricing(TypedDict):
    input: float
    output: float
    context: int


class ModelPricingOverride(TypedDict, total=False):
    input: float
    output: float
    context: int


class PricingConfigError(ValueError):
    """Raised when pricing configuration is invalid."""

    def __init__(
        self,
        *,
        source: str,
        model_id: str,
        field: str,
        value: object,
        detail: str,
    ) -> None:
        self.source = source
        self.model_id = model_id
        self.field = field
        self.value = value
        super().__init__(
            f"Invalid pricing config in {source} for '{model_id}': {field} {detail} (got {value!r})"
        )


def _coerce_nonnegative_float(*, source: str, model_id: str, field: str, value: object) -> float:
    if value is None:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be provided as a non-negative number",
        )
    if isinstance(value, bool):
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be a non-negative number (not boolean)",
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be provided as a non-negative number",
        ) from exc
    if parsed < 0:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be non-negative",
        )
    return parsed


def _coerce_positive_int(*, source: str, model_id: str, field: str, value: object) -> int:
    if value is None:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be provided as a positive integer",
        )
    if isinstance(value, bool):
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be a positive integer (not boolean)",
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise PricingConfigError(
                source=source,
                model_id=model_id,
                field=field,
                value=value,
                detail="must be an integer (no fractional tokens)",
            )
        parsed = int(value)
    elif isinstance(value, str):
        trimmed = value.strip()
        try:
            parsed_float = float(trimmed)
        except ValueError as exc:
            raise PricingConfigError(
                source=source,
                model_id=model_id,
                field=field,
                value=value,
                detail="must be a positive integer",
            ) from exc
        if not parsed_float.is_integer():
            raise PricingConfigError(
                source=source,
                model_id=model_id,
                field=field,
                value=value,
                detail="must be an integer (no fractional tokens)",
            )
        parsed = int(parsed_float)
    else:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be a positive integer",
        )

    if parsed <= 0:
        raise PricingConfigError(
            source=source,
            model_id=model_id,
            field=field,
            value=value,
            detail="must be positive",
        )
    return parsed


class PricingNotFoundError(ValueError):
    """Raised when pricing information is not available for a model.

    This error indicates a configuration gap that should be addressed by
    updating pricing.yaml or adding an override in ~/.ember/config.yaml.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(
            f"No pricing data for model '{model_id}'. "
            "Add pricing to pricing.yaml or configure a pricing override in "
            "~/.ember/config.yaml under models.overrides.<provider>:<id>.pricing."
        )


class Pricing:
    """Load and query pricing information for supported models.

    Prices are quoted per 1,000,000 tokens to minimize floating-point drift.
    Overrides are sourced exclusively from ~/.ember/config.yaml to maintain
    a single, auditable configuration source.

    Attributes:
        yaml_path: Path to the pricing YAML file used for initialization.
        strict: If True, raise PricingNotFoundError for unknown models.
            If False, return zero pricing with a warning (legacy behavior).
    """

    def __init__(
        self,
        yaml_path: Optional[Path] = None,
        strict: bool = True,
    ):
        """Initialize the pricing manager.

        Args:
            yaml_path: Path to pricing.yaml. Defaults to bundled file.
            strict: If True, fail loudly for unknown models. Defaults to True.
        """
        self.yaml_path = yaml_path or Path(__file__).parent / "pricing.yaml"
        self.strict = strict
        self._data = self._load_yaml()
        self._warned_models: set[str] = set()

    @staticmethod
    def _normalize_id(model_id: str) -> str:
        """Normalize model identifiers for fuzzy matching.

        Args:
            model_id: Raw model identifier supplied by a caller.

        Returns:
            Lowercase alphanumeric identifier used for lookups.
        """
        return "".join(ch.lower() for ch in model_id if ch.isalnum())

    def _canonical_model_id(self, model_id: str) -> Optional[str]:
        """Resolve a caller-provided identifier to the canonical form.

        Args:
            model_id: Identifier supplied by configuration or callers.

        Returns:
            Canonical identifier from the pricing table, or None if not found.
        """
        target = self._normalize_id(model_id)
        for existing in self._data.keys():
            if self._normalize_id(existing) == target:
                return existing
        return None

    def _load_yaml(self) -> Dict[str, ModelPricing]:
        """Load the base pricing table from disk.

        Returns:
            Mapping from canonical model IDs to pricing data.

        Raises:
            FileNotFoundError: If pricing.yaml does not exist.
        """
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"Pricing configuration not found: {self.yaml_path}. "
                f"Ensure the Ember package is installed correctly."
            )

        with open(self.yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, Mapping):
            raise TypeError(f"Pricing file {self.yaml_path} must be a mapping at the top level")

        providers = raw.get("providers", {})
        if not isinstance(providers, Mapping):
            raise TypeError(f"Pricing file {self.yaml_path} must contain 'providers' as a mapping")

        source = str(self.yaml_path)
        result: Dict[str, ModelPricing] = {}
        for provider_data in providers.values():
            if not isinstance(provider_data, Mapping):
                raise TypeError(
                    f"Pricing file {self.yaml_path} providers entries must be mappings, "
                    f"got {type(provider_data)!r}"
                )
            models = provider_data.get("models", {})
            if not isinstance(models, Mapping):
                raise TypeError(
                    f"Pricing file {self.yaml_path} provider models must be a mapping, "
                    f"got {type(models)!r}"
                )
            for model_id_raw, costs in models.items():
                model_id = str(model_id_raw)
                if not isinstance(costs, Mapping):
                    raise TypeError(
                        f"Pricing file {self.yaml_path} model '{model_id}' entry must be a mapping"
                    )
                result[model_id] = {
                    "input": _coerce_nonnegative_float(
                        source=source,
                        model_id=model_id,
                        field="input",
                        value=costs.get("input"),
                    ),
                    "output": _coerce_nonnegative_float(
                        source=source,
                        model_id=model_id,
                        field="output",
                        value=costs.get("output"),
                    ),
                    "context": _coerce_positive_int(
                        source=source,
                        model_id=model_id,
                        field="context",
                        value=costs.get("context"),
                    ),
                }

        logger.debug("Loaded pricing for %d models from %s", len(result), self.yaml_path)
        return result

    def _config_overrides(self) -> Dict[str, ModelPricingOverride]:
        """Read pricing overrides from ~/.ember/config.yaml.

        Overrides are declared under models.overrides.<model_id>.pricing with
        fields: input, output, context (all per 1M tokens).

        Returns:
            Mapping from model IDs to override pricing dictionaries.
        """
        overrides: Dict[str, ModelPricingOverride] = {}
        model_overrides = get_model_overrides()

        for key, spec in model_overrides.items():
            pricing_payload = spec.get("pricing")
            if pricing_payload is None:
                continue
            if not isinstance(pricing_payload, Mapping):
                raise PricingConfigError(
                    source="~/.ember/config.yaml",
                    model_id=str(key),
                    field="pricing",
                    value=pricing_payload,
                    detail="must be a mapping",
                )

            try:
                _, model_id = ModelKey.split(key)
            except ValueError:
                model_id = key

            fields: ModelPricingOverride = {}

            if "input" in pricing_payload and "input_per_million" in pricing_payload:
                duplicate = {
                    "input": pricing_payload["input"],
                    "input_per_million": pricing_payload["input_per_million"],
                }
                raise PricingConfigError(
                    source="~/.ember/config.yaml",
                    model_id=str(key),
                    field="pricing.input",
                    value=duplicate,
                    detail="must not set both input and input_per_million",
                )

            # Support both "input" and "input_per_million" for clarity
            for input_key in ("input", "input_per_million"):
                if input_key in pricing_payload:
                    fields["input"] = _coerce_nonnegative_float(
                        source="~/.ember/config.yaml",
                        model_id=str(key),
                        field=f"pricing.{input_key}",
                        value=pricing_payload[input_key],
                    )
                    break

            if "output" in pricing_payload and "output_per_million" in pricing_payload:
                duplicate = {
                    "output": pricing_payload["output"],
                    "output_per_million": pricing_payload["output_per_million"],
                }
                raise PricingConfigError(
                    source="~/.ember/config.yaml",
                    model_id=str(key),
                    field="pricing.output",
                    value=duplicate,
                    detail="must not set both output and output_per_million",
                )

            for output_key in ("output", "output_per_million"):
                if output_key in pricing_payload:
                    fields["output"] = _coerce_nonnegative_float(
                        source="~/.ember/config.yaml",
                        model_id=str(key),
                        field=f"pricing.{output_key}",
                        value=pricing_payload[output_key],
                    )
                    break

            if "context" in pricing_payload:
                fields["context"] = _coerce_positive_int(
                    source="~/.ember/config.yaml",
                    model_id=str(key),
                    field="pricing.context",
                    value=pricing_payload["context"],
                )

            if fields:
                overrides.setdefault(model_id, {}).update(fields)
                logger.debug(
                    "Applied pricing override for %s from config: %s",
                    model_id,
                    fields,
                )

        return overrides

    def get_all_pricing(self) -> Dict[str, ModelPricing]:
        """Return the merged pricing table with config overrides applied.

        Override sources (in precedence order):
            1. Base pricing.yaml data
            2. ~/.ember/config.yaml models.overrides pricing sections

        Returns:
            Pricing data keyed by canonical model identifier.
        """
        merged: Dict[str, ModelPricing] = {
            model_id: {
                "input": pricing["input"],
                "output": pricing["output"],
                "context": pricing["context"],
            }
            for model_id, pricing in self._data.items()
        }

        for model_id, costs in self._config_overrides().items():
            if model_id in merged:
                if "input" in costs:
                    merged[model_id]["input"] = costs["input"]
                if "output" in costs:
                    merged[model_id]["output"] = costs["output"]
                if "context" in costs:
                    merged[model_id]["context"] = costs["context"]
            else:
                # New model from config override
                base: ModelPricing = {"input": 0.0, "output": 0.0, "context": 4096}
                if "input" in costs:
                    base["input"] = costs["input"]
                if "output" in costs:
                    base["output"] = costs["output"]
                if "context" in costs:
                    base["context"] = costs["context"]
                merged[model_id] = base

        return merged

    def has_pricing(self, model_id: str) -> bool:
        """Check if pricing data exists for a model.

        Args:
            model_id: Model identifier to check.

        Returns:
            True if pricing is available, False otherwise.
        """
        all_pricing = self.get_all_pricing()

        if model_id in all_pricing:
            return True

        if "/" in model_id:
            _, remainder = model_id.split("/", 1)
            if remainder in all_pricing:
                return True
            canonical_remainder = self._canonical_model_id(remainder)
            if canonical_remainder and canonical_remainder in all_pricing:
                return True

        canonical = self._canonical_model_id(model_id)
        return bool(canonical and canonical in all_pricing)

    def get_model_pricing(
        self,
        model_id: str,
        strict: Optional[bool] = None,
    ) -> ModelPricing:
        """Return merged pricing for a single model identifier.

        Args:
            model_id: Identifier supplied by the caller.
            strict: Override instance-level strict setting for this call.

        Returns:
            Pricing fields: input, output, context (per 1M tokens).

        Raises:
            PricingNotFoundError: If strict=True and model not found.
        """
        use_strict = strict if strict is not None else self.strict
        all_pricing = self.get_all_pricing()

        # Try exact match first
        if model_id in all_pricing:
            return all_pricing[model_id]

        # Try stripping provider prefix (provider/model_id)
        if "/" in model_id:
            _, remainder = model_id.split("/", 1)
            if remainder in all_pricing:
                return all_pricing[remainder]
            canonical_remainder = self._canonical_model_id(remainder)
            if canonical_remainder and canonical_remainder in all_pricing:
                return all_pricing[canonical_remainder]

        # Try canonical matching
        canonical = self._canonical_model_id(model_id)
        if canonical and canonical in all_pricing:
            return all_pricing[canonical]

        # Model not found
        if use_strict:
            raise PricingNotFoundError(model_id)

        # Legacy fallback with warning (only warn once per model)
        if model_id not in self._warned_models:
            logger.warning(
                "No pricing data for model '%s'. Returning zero cost. "
                "Add pricing to pricing.yaml or ~/.ember/config.yaml",
                model_id,
            )
            self._warned_models.add(model_id)

        return {"input": 0.0, "output": 0.0, "context": 4096}

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        strict: Optional[bool] = None,
    ) -> float:
        """Estimate the USD cost for a single request.

        Args:
            model_id: Identifier of the model being invoked.
            input_tokens: Number of prompt tokens.
            output_tokens: Number of completion tokens.
            strict: Override instance-level strict setting.

        Returns:
            Total cost rounded to six decimal places.

        Raises:
            PricingNotFoundError: If strict=True and model not found.
        """
        pricing = self.get_model_pricing(model_id, strict=strict)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def list_models(self) -> list[str]:
        """Return canonical model identifiers with pricing entries.

        Returns:
            Sorted list of model identifiers.
        """
        return sorted(self.get_all_pricing().keys())


# Global instance defaults to strict=True (fail-closed).
_pricing = Pricing(strict=True)


def get_model_cost(model_id: str, strict: bool | None = None) -> ModelPricing:
    """Return pricing fields for a specific model.

    Args:
        model_id: Identifier of the model being queried.
        strict: If True, raise PricingNotFoundError for unknown models. Defaults to ``None``
            to respect the global Pricing instance's strictness.

    Returns:
        Pricing fields: input, output, context (per 1M tokens).

    Raises:
        PricingNotFoundError: If strict=True and model not found.
    """
    pricing = _pricing.get_model_pricing(model_id, strict=strict)
    return {
        "input": pricing["input"],
        "output": pricing["output"],
        "context": pricing["context"],
    }


def get_model_costs() -> Dict[str, ModelPricing]:
    """Return pricing data for every known model.

    Returns:
        Mapping from canonical model IDs to pricing dictionaries.
    """
    per_million = _pricing.get_all_pricing()
    return {
        model_id: {
            "input": pricing["input"],
            "output": pricing["output"],
            "context": pricing["context"],
        }
        for model_id, pricing in per_million.items()
    }


def get_model_pricing(model_id: str, strict: bool | None = None) -> Tuple[float, float]:
    """Return per-million input and output prices for a model.

    Args:
        model_id: Identifier of the model being queried.
        strict: If True, raise PricingNotFoundError for unknown models. Defaults to ``None``
            to respect the global Pricing instance's strictness.

    Returns:
        Tuple of (input_cost, output_cost) per 1,000,000 tokens.

    Raises:
        PricingNotFoundError: If strict=True and model not found.
    """
    cost = get_model_cost(model_id, strict=strict)
    return cost["input"], cost["output"]
