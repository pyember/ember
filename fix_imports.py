#!/usr/bin/env python3
"""
Fix imports in test files to use the 'ember' package instead of 'src.ember'.

This script replaces all instances of 'from src.ember' or 'import src.ember' with
'from ember' or 'import ember' in Python files under the tests directory.
"""

import os
import re
import glob
from pathlib import Path


def fix_imports_in_file(filepath: str) -> bool:
    """
    Fix imports in a single file by replacing 'src.ember' with 'ember'.

    Args:
        filepath: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Replace imports
    new_content = re.sub(r"from src\.ember", "from ember", content)
    new_content = re.sub(r"import src\.ember", "import ember", new_content)

    # Also fix patch paths
    new_content = re.sub(r'@patch\(\s*"src\.ember', '@patch(\n    "ember', new_content)

    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Fixed imports in {filepath}")
        return True
    return False


def fix_imports_recursively(directory: str) -> int:
    """
    Recursively fix imports in all Python files in a directory.

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
                if fix_imports_in_file(filepath):
                    count += 1
    return count


def main():
    """Main function."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return

    count = fix_imports_recursively(str(tests_dir))
    print(f"Fixed imports in {count} files")


if __name__ == "__main__":
    main()
