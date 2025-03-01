#!/usr/bin/env python3
"""
Fix imports in test files to use the 'ember' package instead of 'src.ember'.

This script replaces all instances of 'from src.ember' or 'import src.ember' with
'from ember' or 'import ember' in Python files under the tests directory.
"""

import os
import re
import glob
import sys
import subprocess
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


def setup_package():
    """Set up the proper package structure for the Ember project."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    print(f"Setting up package structure in {project_root}")
    
    # Check if pytest is installed
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing pytest and pytest-asyncio...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"], check=True)
    
    # Install package in development mode
    print("Installing package in development mode...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Update pytest.ini
    with open(project_root / "pytest.ini", "w") as f:
        f.write("""[pytest]
markers =
    performance: mark test as a performance test.
    integration: mark test as an integration test requiring external dependencies.
pythonpath = 
    .
    src
addopts = --import-mode=importlib
filterwarnings =
    ignore::DeprecationWarning:pkg_resources:3154
    ignore:The configuration option:pytest.PytestDeprecationWarning

# Set default fixture scope to function to avoid asyncio warning
asyncio_default_fixture_loop_scope = function

# Integration Test Instructions:
# To run integration tests, use the following command:
# RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration -v
#
# To run tests that make actual API calls:
# RUN_INTEGRATION_TESTS=1 ALLOW_EXTERNAL_API_CALLS=1 python -m pytest tests/integration -v 
""")
    print("Updated pytest.ini")


def main():
    """Main function."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    tests_dir = project_root / "tests"
    src_dir = project_root / "src"

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return

    if not src_dir.exists():
        print(f"Src directory not found: {src_dir}")
        return

    # First, set up the package structure
    setup_package()
    
    # Then fix imports in the test files
    tests_count = fix_imports_recursively(str(tests_dir))
    print(f"Fixed imports in {tests_count} test files")
    
    # Also fix imports in the src files
    src_count = fix_imports_recursively(str(src_dir))
    print(f"Fixed imports in {src_count} source files")
    
    print("\nSetup complete! You should now be able to run tests with:")
    print("python -m pytest tests")


if __name__ == "__main__":
    main()
