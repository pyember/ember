"""
Configuration for XCS transforms tests.

This module sets up performance test options and other fixtures shared across
transform test modules.
"""

import os

# Note: The --run-perf-tests option is now defined in the root conftest.py


# Setup test environment for all mesh tests
def pytest_configure(config):
    """Configure test environment for XCS transform tests."""
    # Enable test mode for mesh tests
    os.environ["_TEST_MODE"] = "1"


def pytest_unconfigure(config):
    """Clean up after tests."""
    # Clean up test mode flag
    if "_TEST_MODE" in os.environ:
        del os.environ["_TEST_MODE"]
