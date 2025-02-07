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
from ember.registry.dataset.base.models import (
    DatasetEntry,
    DatasetInfo,
    TaskType,
)
from ember.registry.dataset.base.loaders import (
    IDatasetLoader,
    HuggingFaceDatasetLoader,
)
from ember.registry.dataset.base.validators import IDatasetValidator, DatasetValidator
from ember.registry.dataset.base.samplers import IDatasetSampler, DatasetSampler
from ember.registry.dataset.base.transformers import IDatasetTransformer
from ember.registry.dataset.base.preppers import IDatasetPrepper

# If these exist in your “initialization.py” or similar:
from ember.registry.dataset.registry.metadata_registry import DatasetMetadataRegistry
from ember.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from ember.registry.dataset.registry.initialization import DatasetService

# The actual Prepper implementations
from ember.registry.dataset.datasets.short_answer import ShortAnswerPrepper
from ember.registry.dataset.datasets.code_prepper import CodePrepper
