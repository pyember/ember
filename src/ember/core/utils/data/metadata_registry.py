"""Dataset metadata registry.

This module is deprecated in favor of the unified registry system.
Import from ember.core.utils.data.registry instead.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Type

from ember.core.utils.data.base.models import DatasetInfo
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.registry import UNIFIED_REGISTRY, initialize_registry
from ember.core.utils.data.registry import register as unified_register

# Removed explicit warning since this is an internal module
# The warning will be shown just by importing, so the importing code should handle this

# For potential compatibility with internal code
# TODO: Since there are no external users yet, we should update all internal references
# to use ember.core.utils.data.registry directly and eventually remove this compatibility layer.
# See uses in:
# - ember/src/ember/api/data/service.py
# - ember/src/ember/examples/halueval_experiment.py
# - ember/src/ember/examples/mcq_experiment_example.py
# - ember/src/ember/core/utils/data/initialization.py
# - ember/src/ember/core/utils/data/__init__.py
# - ember/src/ember/core/utils/data/service.py
# - ember/tests/unit/core/utils/data/test_metadata_registry.py
# - ember/tests/unit/core/utils/data/test_data_module.py
# - ember/tests/unit/core/utils/data/test_initialization.py
DatasetRegistry = UNIFIED_REGISTRY
register_dataset = unified_register
DatasetRegistryManager = UNIFIED_REGISTRY
DatasetMetadataRegistry = UNIFIED_REGISTRY
