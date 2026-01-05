"""Refresh ``pricing.yaml`` by scraping provider pricing pages with Ember LLMs.

The updater relies on the Ember models API to read provider documentation,
normalize the results, and optionally persist them back to disk.

Examples:
    >>> PricingUpdater().sources["openai"]
    'https://openai.com/api/pricing/'

"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ember.api import models

logger = logging.getLogger(__name__)


class PricingUpdater:
    """Fetch provider pricing pages and update ``pricing.yaml``.

    Attributes:
        pricing_path: Path to the pricing YAML file to read and write.
        sources: Mapping of provider slugs to pricing URLs.

    Examples:
        >>> updater = PricingUpdater()
        >>> sorted(updater.sources)
        ['anthropic', 'google', 'openai']
    """

    def __init__(self, pricing_yaml_path: Optional[Path] = None):
        self.pricing_path = pricing_yaml_path or Path(__file__).parent / "pricing" / "pricing.yaml"
        self.sources = {
            "openai": "https://openai.com/api/pricing/",
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/overview#model-pricing",
            "google": "https://ai.google.dev/gemini-api/docs/pricing",
        }

    def fetch_provider_pricing(self, provider: str, url: str) -> Dict[str, Dict]:
        """Ask an LLM to extract pricing for a provider landing page.

        Args:
            provider: Provider slug such as ``"openai"`` or ``"anthropic"``.
            url: Public pricing page URL to inspect.

        Returns:
            Dict[str, Dict]: Parsed pricing data keyed by model identifier.

        Examples:
            >>> updater = PricingUpdater()
            >>> updater.fetch_provider_pricing(
            ...     "openai", updater.sources["openai"]
            ... )  # doctest: +SKIP
            {'gpt-4': {'input': 30.0, 'output': 60.0, 'context': 128000}}
        """
        prompt = f"""Please visit {url} and extract the current API pricing information.

For each model, provide:
1. Model name/ID exactly as shown
2. Input price per million tokens in USD
3. Output price per million tokens in USD
4. Context window size (if available)

Format your response as YAML like this:
```yaml
model-id-1:
  input: 15.0   # $15 per 1M tokens
  output: 75.0  # $75 per 1M tokens
  context: 200000
model-id-2:
  input: 3.0
  output: 15.0
  context: 200000
```

Only include models that have API pricing (not chat/consumer pricing).
Be precise with model IDs - use exact names from the pricing page."""

        try:
            response = models("claude-3-opus", prompt)

            text = response.text
            yaml_start = text.find("```yaml")
            yaml_end = text.find("```", yaml_start + 7)

            if yaml_start != -1 and yaml_end != -1:
                yaml_content = text[yaml_start + 7 : yaml_end].strip()
                parsed = yaml.safe_load(yaml_content) or {}
                if isinstance(parsed, dict):
                    return parsed

            # Fallback: try parsing the whole response if no fenced YAML was
            # detected or the fenced block was empty.
            parsed = yaml.safe_load(text) or {}
            if isinstance(parsed, dict):
                return parsed
            logger.warning(f"No YAML found in response for {provider}")
            return {}

        except Exception as exc:  # pragma: no cover
            logger.error("Failed to fetch pricing for %s: %s", provider, exc)
            return {}

    def update_pricing(self, dry_run: bool = True) -> Dict:
        """Fetch pricing from all providers and merge into a single mapping.

        Args:
            dry_run: When True, compute and report changes without writing to disk.

        Returns:
            Dict: Complete pricing document that would be persisted.

        Examples:
            >>> PricingUpdater().update_pricing(dry_run=True)['providers'].keys()  # doctest: +SKIP
            dict_keys(['openai', 'anthropic', 'google'])
        """
        logger.info("Starting automated pricing update...")

        if self.pricing_path.exists():
            with open(self.pricing_path) as f:
                current_data = yaml.safe_load(f)
        else:
            current_data = {"version": "1.0", "providers": {}}

        new_data: Dict[str, Dict] = {
            "version": current_data.get("version", "1.0"),
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "providers": {},
        }

        for provider, url in self.sources.items():
            logger.info("Fetching %s pricing from %s...", provider, url)
            models_data = self.fetch_provider_pricing(provider, url)

            if models_data:
                new_data["providers"][provider] = {"source": url, "models": models_data}
                logger.info("Found %d models for %s", len(models_data), provider)
            else:
                existing = current_data.get("providers", {}).get(provider)
                if existing:
                    new_data["providers"][provider] = existing
                    logger.warning("Keeping existing data for %s", provider)

        changes = self._compare_pricing(current_data, new_data)

        if changes:
            logger.info("Found %d pricing changes:", len(changes))
            for change in changes:
                logger.info("  - %s", change)
        else:
            logger.info("No pricing changes detected")

        if not dry_run and changes:
            if self.pricing_path.exists():
                backup_path = self.pricing_path.with_suffix(".yaml.bak")
                self.pricing_path.rename(backup_path)
                logger.info("Backed up to %s", backup_path)

            with open(self.pricing_path, "w") as f:
                yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)
            logger.info("Updated %s", self.pricing_path)

        return new_data

    def _compare_pricing(self, old_data: Dict, new_data: Dict) -> List[str]:
        """Diff two pricing documents and describe the changes.

        Args:
            old_data: Pricing document currently stored on disk.
            new_data: Newly generated pricing document.

        Returns:
            List[str]: Human-readable change descriptions.

        Examples:
            >>> PricingUpdater()._compare_pricing({'providers': {}}, {'providers': {}})
            []
        """
        changes: List[str] = []

        old_providers = old_data.get("providers", {})
        new_providers = new_data.get("providers", {})

        for provider in set(old_providers.keys()) | set(new_providers.keys()):
            old_models = old_providers.get(provider, {}).get("models", {})
            new_models = new_providers.get(provider, {}).get("models", {})

            for model in set(old_models.keys()) | set(new_models.keys()):
                if model not in old_models:
                    changes.append(f"{provider}/{model}: NEW MODEL")
                elif model not in new_models:
                    changes.append(f"{provider}/{model}: REMOVED")
                else:
                    old_m = old_models[model]
                    new_m = new_models[model]

                    if old_m.get("input") != new_m.get("input"):
                        changes.append(
                            f"{provider}/{model}: input ${old_m.get('input')} → "
                            f"${new_m.get('input')}"
                        )
                    if old_m.get("output") != new_m.get("output"):
                        changes.append(
                            f"{provider}/{model}: output ${old_m.get('output')} → "
                            f"${new_m.get('output')}"
                        )

        return changes

    def validate_pricing(self, data: Dict) -> List[str]:
        """Validate the structure of a pricing document.

        Args:
            data: Pricing mapping to validate.

        Returns:
            List[str]: Validation error messages. Empty when the document is valid.

        Examples:
            >>> PricingUpdater().validate_pricing({'providers': {}})
            []
        """
        errors: List[str] = []

        if "providers" not in data:
            errors.append("Missing 'providers' key")
            return errors

        for provider, provider_data in data["providers"].items():
            if "models" not in provider_data:
                errors.append(f"{provider}: Missing 'models' key")
                continue

            for model, pricing in provider_data["models"].items():
                if "input" not in pricing:
                    errors.append(f"{provider}/{model}: Missing 'input' price")
                elif not isinstance(pricing["input"], (int, float)) or pricing["input"] < 0:
                    errors.append(f"{provider}/{model}: Invalid 'input' price")

                if "output" not in pricing:
                    errors.append(f"{provider}/{model}: Missing 'output' price")
                elif not isinstance(pricing["output"], (int, float)) or pricing["output"] < 0:
                    errors.append(f"{provider}/{model}: Invalid 'output' price")

                if "context" in pricing:
                    if not isinstance(pricing["context"], int) or pricing["context"] < 1:
                        errors.append(f"{provider}/{model}: Invalid 'context' size")

        return errors


def update_pricing_cli() -> int:
    """Command-line entry point for pricing refresh and validation.

    Returns:
        int: Process exit code (0 indicates success).

    Examples:
        >>> update_pricing_cli()  # doctest: +SKIP
        0
    """
    import argparse

    parser = argparse.ArgumentParser(description="Update model pricing data")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without updating file")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing pricing.yaml"
    )

    args = parser.parse_args()

    updater = PricingUpdater()

    if args.validate_only:
        with open(updater.pricing_path) as f:
            data = yaml.safe_load(f)

        errors = updater.validate_pricing(data)
        if errors:
            print(f"Found {len(errors)} validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("Pricing data is valid")
        return 0

    try:
        updater.update_pricing(dry_run=args.dry_run)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error updating pricing: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(update_pricing_cli())
