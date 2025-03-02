#!/usr/bin/env python
"""
Fuzzing test suite for Ember using Atheris.

This script discovers and runs all fuzzing tests defined in the fuzzing directory.
"""

import os
import sys
import time
import glob
import importlib.util
import argparse
from typing import List, Optional

# Ensure Ember is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def discover_fuzz_tests() -> List[str]:
    """Discover all fuzzing test modules in the current directory."""
    fuzz_test_files = glob.glob(os.path.join(os.path.dirname(__file__), 'fuzz_*.py'))
    return [os.path.basename(f) for f in fuzz_test_files]


def run_fuzz_test(test_file: str, time_limit: Optional[int] = None) -> bool:
    """Run a single fuzzing test with an optional time limit."""
    module_name = os.path.splitext(test_file)[0]
    
    # Import the module
    spec = importlib.util.spec_from_file_location(
        module_name, 
        os.path.join(os.path.dirname(__file__), test_file)
    )
    if spec is None or spec.loader is None:
        print(f"Failed to load module: {test_file}")
        return False
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check if the module has a run_fuzzer function
    if not hasattr(module, 'run_fuzzer'):
        print(f"Module {test_file} does not have a run_fuzzer function")
        return False
    
    print(f"Running fuzzing test: {test_file}")
    start_time = time.time()
    
    try:
        if time_limit:
            # Run with time limit
            module.run_fuzzer(time_limit=time_limit)
        else:
            # Run without time limit (typically uses default iterations)
            module.run_fuzzer()
        
        elapsed = time.time() - start_time
        print(f"Fuzzing test {test_file} completed in {elapsed:.2f} seconds")
        return True
    
    except Exception as e:
        print(f"Fuzzing test {test_file} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Ember fuzzing tests")
    parser.add_argument("--time_limit", type=int, default=None, 
                      help="Time limit in seconds for each fuzzing test")
    parser.add_argument("--test", type=str, default=None,
                      help="Run a specific test file (e.g. fuzz_parser.py)")
    args = parser.parse_args()
    
    # Create output directory for crash artifacts
    os.makedirs('fuzzing_results', exist_ok=True)
    
    # Discover tests
    if args.test:
        if not args.test.startswith('fuzz_'):
            args.test = f'fuzz_{args.test}'
        if not args.test.endswith('.py'):
            args.test = f'{args.test}.py'
        test_files = [args.test]
    else:
        test_files = discover_fuzz_tests()
    
    if not test_files:
        print("No fuzzing tests discovered!")
        return 1
    
    print(f"Discovered {len(test_files)} fuzzing tests")
    
    # Run tests
    failures = 0
    for test_file in test_files:
        if not run_fuzz_test(test_file, args.time_limit):
            failures += 1
    
    if failures:
        print(f"FAILED: {failures} out of {len(test_files)} fuzzing tests failed")
        return 1
    else:
        print(f"SUCCESS: All {len(test_files)} fuzzing tests passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())