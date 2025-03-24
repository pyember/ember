#!/usr/bin/env python
"""
Migration script for converting to the Ember Exception Architecture.

This script helps with identifying and migrating exceptions in the codebase
to the new exception architecture defined in ember.core.exceptions.

Usage:
  python scripts/migrate_exceptions.py --scan  # Scan for exception usage
  python scripts/migrate_exceptions.py --module path/to/module  # Analyze specific module
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Map of standard exceptions to their Ember counterparts
EXCEPTION_MAPPING = {
    "ValueError": "ValidationError",
    "TypeError": "IncompatibleTypeError",
    "KeyError": "ItemNotFoundError",  # Context-specific, may need to be ConfigValueError
    "NotImplementedError": "NotImplementedFeatureError",
    "RuntimeError": "EmberError",  # Generic fallback, context-specific recommended
    "Exception": "EmberError",
}

# Regex patterns for finding exceptions
RAISE_PATTERN = re.compile(r"raise\s+(\w+)(?:\(|\s)")
EXCEPT_PATTERN = re.compile(r"except\s+(\w+)(?:\s+as\s+\w+)?:")
EXCEPTION_CLASS_PATTERN = re.compile(r"class\s+(\w+)(?:\((\w+)\))?:")


def find_source_files(root_dir: str) -> List[str]:
    """Find all Python source files in the given directory."""
    py_files = []
    for path in Path(root_dir).rglob("*.py"):
        py_files.append(str(path))
    return py_files


def scan_file_for_exceptions(file_path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Scan a file for exception usage and definitions."""
    raised_exceptions = {}
    caught_exceptions = {}
    defined_exceptions = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            
            # Find raised exceptions
            for match in RAISE_PATTERN.finditer(content):
                exception_name = match.group(1)
                raised_exceptions[exception_name] = raised_exceptions.get(exception_name, 0) + 1
            
            # Find caught exceptions
            for match in EXCEPT_PATTERN.finditer(content):
                exception_name = match.group(1)
                caught_exceptions[exception_name] = caught_exceptions.get(exception_name, 0) + 1
            
            # Find exception class definitions
            for match in EXCEPTION_CLASS_PATTERN.finditer(content):
                class_name = match.group(1)
                parent_class = match.group(2) if match.group(2) else "object"
                
                if class_name.endswith("Error") or class_name.endswith("Exception"):
                    defined_exceptions[class_name] = parent_class
        
        except UnicodeDecodeError:
            print(f"Warning: Could not decode {file_path}")
    
    return raised_exceptions, caught_exceptions, defined_exceptions


def scan_codebase(root_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Scan the entire codebase for exception usage."""
    all_raised_exceptions = {}
    all_caught_exceptions = {}
    custom_exceptions = {}
    
    py_files = find_source_files(root_dir)
    for file_path in py_files:
        raised, caught, defined = scan_file_for_exceptions(file_path)
        
        # Update raised exceptions counts
        for exc_name, count in raised.items():
            all_raised_exceptions[exc_name] = all_raised_exceptions.get(exc_name, 0) + count
        
        # Update caught exceptions counts
        for exc_name, count in caught.items():
            all_caught_exceptions[exc_name] = all_caught_exceptions.get(exc_name, 0) + count
        
        # Track custom exception definitions
        for exc_name, parent_class in defined.items():
            if exc_name not in custom_exceptions:
                custom_exceptions[exc_name] = []
            custom_exceptions[exc_name].append(file_path)
    
    return all_raised_exceptions, all_caught_exceptions, custom_exceptions


def analyze_module(module_path: str) -> None:
    """Analyze a specific module for exception usage."""
    if not os.path.exists(module_path):
        print(f"Error: Module path '{module_path}' does not exist")
        return
    
    print(f"\nAnalyzing exceptions in {module_path}:")
    print("=" * 60)
    
    if os.path.isdir(module_path):
        py_files = find_source_files(module_path)
    else:
        py_files = [module_path]
    
    for file_path in py_files:
        raised, caught, defined = scan_file_for_exceptions(file_path)
        
        if not (raised or caught or defined):
            continue
        
        print(f"\nFile: {file_path}")
        
        if defined:
            print("\n  Defined exceptions:")
            for exc_name, parent_class in defined.items():
                suggested = ""
                if parent_class == "Exception" or parent_class == "BaseException":
                    suggested = " -> Consider extending from EmberError"
                print(f"    - {exc_name}({parent_class}){suggested}")
        
        if raised:
            print("\n  Raised exceptions:")
            for exc_name, count in raised.items():
                suggested = ""
                if exc_name in EXCEPTION_MAPPING:
                    suggested = f" -> Consider using {EXCEPTION_MAPPING[exc_name]}"
                print(f"    - {exc_name} ({count} occurrences){suggested}")
        
        if caught:
            print("\n  Caught exceptions:")
            for exc_name, count in caught.items():
                print(f"    - {exc_name} ({count} occurrences)")


def main():
    parser = argparse.ArgumentParser(description='Exception migration tool for Ember')
    parser.add_argument('--scan', action='store_true', 
                        help='Scan the codebase for exception usage')
    parser.add_argument('--module', type=str, 
                        help='Analyze a specific module for exception usage')
    
    args = parser.parse_args()
    
    if args.scan:
        print("Scanning codebase for exception usage...")
        raised, caught, custom = scan_codebase('src/ember')
        
        print("\nSummary of raised exceptions:")
        print("=" * 60)
        for exc_name, count in sorted(raised.items(), key=lambda x: x[1], reverse=True):
            suggested = ""
            if exc_name in EXCEPTION_MAPPING:
                suggested = f" -> Consider using {EXCEPTION_MAPPING[exc_name]}"
            print(f"{exc_name}: {count} occurrences{suggested}")
        
        print("\nCustom exception classes:")
        print("=" * 60)
        for exc_name, file_paths in sorted(custom.items()):
            print(f"{exc_name}: defined in {len(file_paths)} file(s)")
            for path in file_paths[:3]:  # Show up to 3 files
                print(f"  - {path}")
            if len(file_paths) > 3:
                print(f"  - ... and {len(file_paths) - 3} more")
    
    elif args.module:
        analyze_module(args.module)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()