#!/usr/bin/env python3
"""
Script to remove unused imports in Python files.

This script analyzes Python files in the tests directory and removes imports
that are reported as unused by pylint.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Optional


def get_unused_imports(file_path: str) -> Set[str]:
    """
    Run pylint on a file and extract unused imports.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Set of unused import names
    """
    result = subprocess.run(
        ["pylint", "--disable=all", "--enable=unused-import", file_path],
        capture_output=True,
        text=True,
    )

    unused_imports = set()
    for line in result.stdout.splitlines():
        match = re.search(
            r"Unused ([a-zA-Z0-9_]+) imported from ([a-zA-Z0-9_\.]+)", line
        )
        if match:
            import_name = match.group(1)
            unused_imports.add(import_name)

    return unused_imports


def remove_unused_imports(file_path: str, unused_imports: Set[str]) -> bool:
    """
    Remove unused imports from a file.

    Args:
        file_path: Path to the file to fix
        unused_imports: Set of import names to remove

    Returns:
        bool: True if file was modified, False otherwise
    """
    with open(file_path, "r") as f:
        content = f.read()

    modified = False
    lines = content.splitlines()
    new_lines = []

    for line in lines:
        should_keep = True

        # Check for imports on a single line (e.g., from typing import Dict, List, Optional)
        single_line_match = re.match(
            r"^from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.+)$", line
        )
        if single_line_match:
            module = single_line_match.group(1)
            imports = single_line_match.group(2)

            # Handle comma-separated imports
            import_items = [i.strip() for i in imports.split(",")]
            new_imports = []

            for item in import_items:
                # Extract the actual import name (without 'as' clause)
                actual_import = item.split(" as ")[0].strip()
                if actual_import not in unused_imports:
                    new_imports.append(item)

            if not new_imports:
                # All imports on this line are unused, remove the entire line
                should_keep = False
                modified = True
            elif len(new_imports) != len(import_items):
                # Some imports were unused, keep the line but with only the used ones
                new_line = f"from {module} import {', '.join(new_imports)}"
                new_lines.append(new_line)
                should_keep = False
                modified = True

        # Check for single import (e.g., import pytest)
        single_import_match = re.match(
            r"^import\s+([a-zA-Z0-9_\.]+)(\s+as\s+[a-zA-Z0-9_]+)?$", line
        )
        if single_import_match:
            import_name = single_import_match.group(1).split(".")[
                -1
            ]  # Get the last part of the import
            if import_name in unused_imports:
                should_keep = False
                modified = True

        if should_keep:
            new_lines.append(line)

    if modified:
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))
        print(f"Removed unused imports in {file_path}")
        return True

    return False


def fix_imports_recursively(directory: str) -> int:
    """
    Recursively fix unused imports in all Python files in a directory.

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
                unused_imports = get_unused_imports(filepath)
                if unused_imports and remove_unused_imports(filepath, unused_imports):
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
    print(f"Fixed unused imports in {count} files")


if __name__ == "__main__":
    main()
