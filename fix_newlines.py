#!/usr/bin/env python3
"""
Script to add missing newlines at the end of Python files.

This script looks for Python files that are missing a final newline
and adds one to fix C0304: Final newline missing (missing-final-newline) errors.
"""

import os
from pathlib import Path


def fix_newlines_in_file(filepath: str) -> bool:
    """
    Add a missing newline at the end of a file if needed.

    Args:
        filepath: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.endswith("\n"):
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n")
        print(f"Added final newline to {filepath}")
        return True
    return False


def fix_newlines_recursively(directory: str) -> int:
    """
    Recursively fix newlines in all Python files in a directory.

    Args:
        directory: Directory to search for Python files

    Returns:
        int: Number of files modified
    """
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                if fix_newlines_in_file(filepath):
                    count += 1
    return count


def main():
    """Main function."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()

    # Fix both src and tests directories
    tests_dir = project_root / "tests"
    src_dir = project_root / "src"

    total_count = 0
    if tests_dir.exists():
        count = fix_newlines_recursively(str(tests_dir))
        total_count += count
        print(f"Fixed newlines in {count} test files")

    if src_dir.exists():
        count = fix_newlines_recursively(str(src_dir))
        total_count += count
        print(f"Fixed newlines in {count} source files")

    print(f"Fixed newlines in {total_count} files total")


if __name__ == "__main__":
    main()
