#!/usr/bin/env python3
"""Test suite runner for Ember.

This script provides a convenient way to run different test categories
with appropriate configurations.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, env=None):
    """Run a command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Ember test suite")
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "benchmarks",
            "golden",
            "models",
            "data",
            "operators",
            "xcs",
            "thread-safety",
            "error-handling",
            "quick",
            "ci",
        ],
        help="Test category to run",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-x", "--exitfirst", action="store_true", help="Exit on first failure")
    parser.add_argument("-k", "--keyword", help="Run tests matching keyword")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["pytest"]

    # Add verbose flag
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")

    # Add exit on first failure
    if args.exitfirst:
        cmd.append("-x")

    # Add keyword filter
    if args.keyword:
        cmd.extend(["-k", args.keyword])

    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=ember", "--cov-report=html", "--cov-report=term"])

    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Select test paths based on category
    test_paths = []

    if args.category == "all":
        test_paths = ["tests/"]

    elif args.category == "unit":
        test_paths = ["tests/unit/"]

    elif args.category == "integration":
        test_paths = ["tests/integration/"]

    elif args.category == "benchmarks":
        test_paths = ["tests/benchmarks/"]
        cmd.extend(["-m", "benchmark"])

    elif args.category == "golden":
        test_paths = ["tests/golden/"]

    elif args.category == "models":
        test_paths = ["tests/unit/models/", "tests/integration/api/test_models_api.py"]

    elif args.category == "data":
        test_paths = [
            "tests/unit/core/utils/data/",
            "tests/integration/api/test_data_streaming.py",
        ]

    elif args.category == "operators":
        test_paths = [
            "tests/unit/operators/",
            "tests/integration/core/test_operator_integration.py",
        ]

    elif args.category == "xcs":
        test_paths = ["tests/unit/xcs/", "tests/integration/xcs/"]

    elif args.category == "thread-safety":
        test_paths = ["tests/integration/core/test_thread_safety.py"]

    elif args.category == "error-handling":
        test_paths = ["tests/integration/test_error_handling.py"]

    elif args.category == "quick":
        # Quick smoke tests
        test_paths = [
            "tests/unit/core/test_ember_model.py",
            "tests/unit/xcs/test_jit.py",
            "tests/integration/test_api_integration.py",
        ]
        cmd.extend(["--maxfail=3"])

    elif args.category == "ci":
        # CI configuration - skip slow tests
        test_paths = ["tests/"]
        cmd.extend(["-m", "not benchmark", "--maxfail=10", "--tb=short"])

    # Add test paths
    cmd.extend(test_paths)

    # Set up environment
    env = os.environ.copy()

    # Add project root to PYTHONPATH
    project_root = Path(__file__).parent.parent
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(project_root)

    # Set test mode to avoid hitting real APIs
    env["EMBER_TEST_MODE"] = "true"

    # Run tests
    return_code = run_command(cmd, env)

    # Print summary
    print(f"\n{'=' * 60}")
    if return_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
