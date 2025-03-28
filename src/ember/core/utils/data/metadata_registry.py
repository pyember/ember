"""Dataset metadata registry compatibility layer.

This module provides backward compatibility aliases for code still using the legacy registry system.
Import from ember.core.utils.data.registry instead.

WARNING: This module exists only for backward compatibility and will be removed in a future version.
All new code should use ember.core.utils.data.registry directly.
"""

import warnings

from ember.core.utils.data.registry import DATASET_REGISTRY
from ember.core.utils.data.registry import register as dataset_register

# Issue a warning when importing this module
warnings.warn(
    "The metadata_registry module is deprecated. "
    "Use ember.core.utils.data.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Compatibility aliases for internal code
# These aliases are provided to maintain backward compatibility with existing code
DatasetRegistry = DATASET_REGISTRY
register_dataset = dataset_register
DatasetRegistryManager = DATASET_REGISTRY
DatasetMetadataRegistry = DATASET_REGISTRY
