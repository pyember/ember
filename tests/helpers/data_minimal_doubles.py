"""
Minimal test doubles for data processing components.

This module provides simplified test doubles that implement just enough functionality
to test client code without duplicating the implementation. Following the
principle of "avoid overmocking" from CLAUDE.md guidelines.

Also includes minimal doubles for the unified data registry system.
"""

import json
import logging
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

# Setup logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
U = TypeVar("U")


class MinimalLoader:
    """Base class for data loaders."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> Any:
        """Load data from file."""
        raise NotImplementedError("Subclasses must implement load method")


class MinimalJsonLoader(MinimalLoader):
    """Load data from JSON files."""

    def load(self) -> List[Dict[str, Any]]:
        """Load JSON data."""
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return data


class MinimalTextLoader(MinimalLoader):
    """Load data from text files."""

    def load(self) -> List[str]:
        """Load text data line by line."""
        with open(self.file_path, "r") as file:
            data = [line.strip() for line in file]
        return data


class MinimalCsvLoader(MinimalLoader):
    """Load data from CSV files."""

    def __init__(self, file_path: str, delimiter: str = ","):
        """Initialize with file path and delimiter."""
        super().__init__(file_path)
        self.delimiter = delimiter

    def load(self) -> List[Dict[str, Any]]:
        """Load CSV data."""
        results = []
        with open(self.file_path, "r") as file:
            lines = [line.strip() for line in file]
            if not lines:
                return []

            headers = lines[0].split(self.delimiter)
            for i in range(1, len(lines)):
                values = lines[i].split(self.delimiter)
                if len(values) == len(headers):
                    row = dict(zip(headers, values))
                    results.append(row)
        return results


class MinimalTransformer(Generic[T, U]):
    """Base class for data transformers."""

    def transform(self, data: T) -> U:
        """Transform data."""
        raise NotImplementedError("Subclasses must implement transform method")


class MinimalFilter(MinimalTransformer[List[T], List[T]]):
    """Filter data based on a predicate."""

    def __init__(self, predicate: Callable[[T], bool]):
        """Initialize with predicate function."""
        self.predicate = predicate

    def transform(self, data: List[T]) -> List[T]:
        """Filter data using predicate."""
        return [item for item in data if self.predicate(item)]


class MinimalShuffler(MinimalTransformer[List[T], List[T]]):
    """Shuffle data randomly."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with random seed."""
        self.seed = seed
        if seed is not None:
            import random

            self._rng = random.Random(seed)
        else:
            import random

            self._rng = random.Random()

    def transform(self, data: List[T]) -> List[T]:
        """Shuffle data."""
        result = data.copy()
        self._rng.shuffle(result)
        return result


class MinimalTokenizer(MinimalTransformer[List[str], List[List[str]]]):
    """Tokenize text data."""

    def __init__(self, tokenize_fn: Callable[[str], List[str]]):
        """Initialize with tokenization function."""
        self.tokenize_fn = tokenize_fn

    def transform(self, data: List[str]) -> List[List[str]]:
        """Tokenize data."""
        return [self.tokenize_fn(item) for item in data]


class MinimalSampler:
    """Base class for data sampling."""

    def sample(self, data: List[T]) -> List[T]:
        """Sample data."""
        raise NotImplementedError("Subclasses must implement sample method")


class MinimalRandomSampler(MinimalSampler):
    """Random sampling of data."""

    def __init__(self, sample_size: int, seed: Optional[int] = None):
        """Initialize with sample size and random seed."""
        self.sample_size = sample_size
        self.seed = seed
        if seed is not None:
            import random

            self._rng = random.Random(seed)
        else:
            import random

            self._rng = random.Random()

    def sample(self, data: List[T]) -> List[T]:
        """Sample data randomly."""
        if len(data) <= self.sample_size:
            return data.copy()
        return self._rng.sample(data, self.sample_size)


class MinimalValidator:
    """Base class for data validation."""

    def validate(self, data: Any) -> bool:
        """Validate data."""
        raise NotImplementedError("Subclasses must implement validate method")


class MinimalSchemaValidator(MinimalValidator):
    """Validate data against a schema."""

    def __init__(self, schema: Dict[str, Any]):
        """Initialize with schema definition."""
        self.schema = schema

    def validate(self, data: Any) -> bool:
        """Validate data against schema.

        This is a very simplified implementation that just checks
        for required fields and basic types.
        """
        if not isinstance(data, dict):
            return False

        if "required" in self.schema:
            for field in self.schema["required"]:
                if field not in data:
                    return False

        if "properties" in self.schema:
            for field, props in self.schema["properties"].items():
                if field in data and "type" in props:
                    if props["type"] == "string" and not isinstance(data[field], str):
                        return False
                    elif props["type"] == "integer" and not isinstance(
                        data[field], int
                    ):
                        return False
                    elif props["type"] == "number" and not isinstance(
                        data[field], (int, float)
                    ):
                        return False
                    elif props["type"] == "boolean" and not isinstance(
                        data[field], bool
                    ):
                        return False
                    elif props["type"] == "array" and not isinstance(data[field], list):
                        return False
                    elif props["type"] == "object" and not isinstance(
                        data[field], dict
                    ):
                        return False

        return True


class MinimalDataService:
    """Orchestrates data processing."""

    def __init__(self):
        """Initialize with empty components."""
        self.loader = None
        self.validators = []
        self.transformers = []
        self.sampler = None

    def set_loader(self, loader: MinimalLoader) -> None:
        """Set the data loader."""
        self.loader = loader

    def add_validator(self, validator: MinimalValidator) -> None:
        """Add a data validator."""
        self.validators.append(validator)

    def add_transformer(self, transformer: MinimalTransformer) -> None:
        """Add a data transformer."""
        self.transformers.append(transformer)

    def set_sampler(self, sampler: MinimalSampler) -> None:
        """Set the data sampler."""
        self.sampler = sampler

    def load_and_process(self) -> Any:
        """Load and process data."""
        if self.loader is None:
            raise ValueError("Loader not set")

        # Load data
        data = self.loader.load()

        # Validate data
        if self.validators:
            if isinstance(data, list):
                for item in data:
                    for validator in self.validators:
                        if not validator.validate(item):
                            raise ValueError(f"Validation failed for item: {item}")
            else:
                for validator in self.validators:
                    if not validator.validate(data):
                        raise ValueError("Validation failed for data")

        # Transform data
        for transformer in self.transformers:
            data = transformer.transform(data)

        # Sample data
        if self.sampler is not None and isinstance(data, list):
            data = self.sampler.sample(data)

        return data


# Export minimal test doubles
__all__ = [
    "MinimalLoader",
    "MinimalJsonLoader",
    "MinimalTextLoader",
    "MinimalCsvLoader",
    "MinimalTransformer",
    "MinimalFilter",
    "MinimalShuffler",
    "MinimalTokenizer",
    "MinimalSampler",
    "MinimalRandomSampler",
    "MinimalValidator",
    "MinimalSchemaValidator",
    "MinimalDataService",
]
