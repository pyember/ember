#!/usr/bin/env python3
"""
Script to fix imports in test files by adding try/except blocks for both import paths.
"""

import os
import re
import glob

# Pattern to match ember.core imports
IMPORT_PATTERN = r"from ember\.core\.utils\.eval\.([\w\.]+) import ([\w\s,\(\)\n]+)"

# Replacement template
REPLACEMENT_TEMPLATE = """try:
    from ember.core.utils.eval.{module} import {imports}
except ImportError:
    from src.ember.core.utils.eval.{module} import {imports}"""


def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Add debug imports at the top if not already present
    if "import sys" not in content and "import os" not in content:
        debug_imports = """import sys
import os
"""
        # Find the first import statement
        first_import = re.search(r"^import .*$", content, re.MULTILINE)
        if first_import:
            pos = first_import.start()
            content = content[:pos] + debug_imports + content[pos:]

    # Add debug print statements if not already present
    if 'print(f"Python path:' not in content:
        debug_prints = """
# Print current path for debugging
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
"""
        # Find the position after imports
        imports_end = 0
        for match in re.finditer(r"^(?:import|from) .*$", content, re.MULTILINE):
            imports_end = max(imports_end, match.end())

        if imports_end > 0:
            content = content[:imports_end] + debug_prints + content[imports_end:]

    # Replace ember.core imports with try/except blocks
    def replace_import(match):
        module = match.group(1)
        imports = match.group(2)
        return REPLACEMENT_TEMPLATE.format(module=module, imports=imports)

    modified_content = re.sub(IMPORT_PATTERN, replace_import, content)

    if content != modified_content:
        with open(file_path, "w") as f:
            f.write(modified_content)
        print(f"Fixed imports in {file_path}")
    else:
        print(f"No changes needed in {file_path}")


def main():
    """Find and fix imports in all test files."""
    test_files = glob.glob("tests/unit/core/utils/eval/test_*.py")
    for file_path in test_files:
        fix_imports_in_file(file_path)


if __name__ == "__main__":
    main()
