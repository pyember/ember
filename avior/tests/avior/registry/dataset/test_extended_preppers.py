"""
test_extended_preppers.py

A comprehensive test suite for extended dataset tasks, including ShortAnswer and Code tasks.
Demonstrates best practices reminiscent of Google L6–L9+ SWE IC level, focusing on:
  - SOLID design testing
  - Comprehensive parameterization
  - Mocking external dependencies
  - Integration coverage with DatasetService

Modules under test:
  - ShortAnswerPrepper
  - CodePrepper
  - ShortAnswerEvaluator
  - CodeEvaluator
  - DatasetService integration with short-answer and code tasks
"""

import os
import pytest
import logging
import subprocess
from unittest.mock import patch, MagicMock

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datasets import Dataset
from dataclasses import dataclass

# --------------------------------------------
# UPDATED IMPORTS
# --------------------------------------------

# Base / registry imports
from avior.registry.dataset.base.models import (
    DatasetEntry,
    DatasetInfo,
    TaskType,
)
from avior.registry.dataset.base.loaders import (
    IDatasetLoader,
    HuggingFaceDatasetLoader,
)
from avior.registry.dataset.base.validators import (
    IDatasetValidator,
    DatasetValidator
)
from avior.registry.dataset.base.samplers import (
    IDatasetSampler,
    DatasetSampler
)
from avior.registry.dataset.base.transformers import IDatasetTransformer
from avior.registry.dataset.base.preppers import IDatasetPrepper

# If these exist in your “initialization.py” or similar:
from avior.registry.dataset.registry.metadata_registry import DatasetMetadataRegistry
from avior.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from avior.registry.dataset.registry.initialization import DatasetService

# The actual Prepper implementations
from avior.registry.dataset.datasets.short_answer import ShortAnswerPrepper
from avior.registry.dataset.datasets.code_prepper import CodePrepper