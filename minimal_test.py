#!/usr/bin/env python3
"""
Minimal test of the discovery service with no frills.
"""

import logging
import time
import os

# Turn off all logging
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ["httpx", "httpcore", "openai", "ember"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Import directly what we need
from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService

# Create service 
service = ModelDiscoveryService()

# Time the discovery
start = time.time()
print("Starting discovery...")
models = service.discover_models()
duration = time.time() - start

# Report results
print(f"\nSuccess! Found {len(models)} models in {duration:.2f} seconds")
print("First 5 models:")
for i, model_id in enumerate(list(models.keys())[:5]):
    print(f"  - {model_id}")