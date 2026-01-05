import pytest

from ember._internal.exceptions import ConfigValueError
from ember.models import catalog


def test_override_mapping_requires_mapping_values() -> None:
    with pytest.raises(ConfigValueError, match="Override entries must be mappings"):
        catalog._normalize_override_mapping({"openai": "not-a-mapping"})  # type: ignore[arg-type]


def test_override_entry_requires_provider_when_unqualified() -> None:
    with pytest.raises(ConfigValueError, match="must include provider"):
        catalog._normalize_override_mapping({"gpt-4": {"hidden": True}})


def test_override_entry_rejects_non_bool_hidden() -> None:
    with pytest.raises(ConfigValueError, match="hidden must be a boolean"):
        catalog._normalize_override_mapping({"openai": {"gpt-4": {"hidden": "true"}}})

